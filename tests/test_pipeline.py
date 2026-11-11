import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def loader(count=4, batch_size=2, drop_last=False):
    return DataLoader(TensorDataset(torch.ones(count, 3, 1), torch.ones(count, 1)),
                      batch_size=batch_size, drop_last=drop_last)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.moves = []

    def to(self, *args, **kwargs):
        self.moves.append(str(args[0]))
        return super().to(*args, **kwargs)

    def forward(self, x):
        return x[:, -1, :] * self.weight


@pytest.mark.parametrize("header", ["Date,Close", "date,close", " DATE , CLOSE "])
def test_csv_normalizes_before_date_parse(pipeline, tmp_path, header):
    path = tmp_path / "prices.csv"
    path.write_text(header + "\n2026-01-02,11\n2026-01-01,10\n", encoding="utf-8")
    frame = pipeline.preprocess.load_and_filter_data(path)
    assert frame.close.tolist() == [10, 11]
    assert frame.index.is_monotonic_increasing
    assert isinstance(frame.index, pd.DatetimeIndex)


@pytest.mark.parametrize("body", [
    "date,close\n2026-01-01,1\n2026-01-01,2\n",
    "date,close\n2026-01-01,inf\n",
    "date,close\ninvalid-date,1\n",
    "date,close\n2026-01-01,not-a-price\n",
    "Date,date,Close\n2026-01-01,2026-01-02,1\n",
    "date,close\n,\n",
])
def test_csv_rejects_ambiguous_or_invalid_data(pipeline, tmp_path, body):
    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.preprocess.load_and_filter_data(path)


@pytest.mark.parametrize("ratios", [(0, .1), (.8, .2), (.9, .2), (float('nan'), .1), (.8, 0)])
def test_invalid_ratios(pipeline, ratios):
    df = pd.DataFrame({'close': np.arange(100)})
    with pytest.raises(ValueError):
        pipeline.preprocess.train_val_test_split(df, *ratios)


def test_split_is_nonempty_disjoint_and_chronological(pipeline):
    df = pd.DataFrame({'close': np.arange(100)}, index=pd.date_range('2026-01-01', periods=100))
    a, b, c = pipeline.preprocess.train_val_test_split(df, .6, .2)
    assert [len(x) for x in (a,b,c)] == [60,20,20]
    assert a.index[-1] < b.index[0] < c.index[0]
    with pytest.raises(ValueError):
        pipeline.preprocess.train_val_test_split(df.iloc[:2], .6, .2)
    with pytest.raises(ValueError):
        pipeline.preprocess.train_val_test_split(df.iloc[::-1])


@pytest.mark.parametrize("window,horizon", [(0,1),(3,0),(-2,1),(True,1),(2,1.5)])
def test_invalid_sequence_parameters(pipeline, window, horizon):
    with pytest.raises(ValueError):
        pipeline.preprocess.create_sequences(np.arange(10), window, horizon)


def test_sequence_alignment(pipeline):
    x, y = pipeline.preprocess.create_sequences(np.arange(10), 3, 2)
    assert x.shape == (6,3,1)
    np.testing.assert_array_equal(x[0,:,0], [0,1,2])
    np.testing.assert_array_equal(y[:,0], [4,5,6,7,8,9])


@pytest.mark.parametrize("values", [np.arange(3), np.array([1,np.nan,3,4]), np.ones((5,1))])
def test_unusable_sequences_fail_early(pipeline, values):
    with pytest.raises(ValueError):
        pipeline.preprocess.create_sequences(values, 3, 1)


def test_scaler_fits_only_training_and_preserves_horizon(pipeline):
    frames = [pd.DataFrame({'close':np.arange(start,start+10,dtype=float)},
                          index=pd.date_range('2026-01-01', periods=30)[i*10:(i+1)*10])
              for i,start in enumerate((10,100,1000))]
    x,y,xv,yv,xt,yt,scaler = pipeline.preprocess.prepare_datasets(*frames, 3, 2)
    np.testing.assert_array_equal(scaler.data_min_,[10])
    np.testing.assert_array_equal(scaler.data_max_,[19])
    assert xv.min() > 1 and xt.min() > 1
    np.testing.assert_allclose(scaler.inverse_transform(y)[:,0],np.arange(14,20))
    assert x.shape==(6,3,1) and yt.shape==(6,1)
    with pytest.raises(ValueError):
        pipeline.preprocess.prepare_datasets(frames[0],frames[0],frames[2],3,2)


def test_model_and_both_training_tensors_are_moved(pipeline, monkeypatch):
    calls=[]
    original=torch.Tensor.to
    def tracked(self,*args,**kwargs):
        calls.append(tuple(self.shape))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',tracked)
    model=TinyModel()
    pipeline.train.train_model(model,loader(2),loader(2),{'epochs':1},'cpu')
    assert model.moves == ['cpu']
    assert calls.count((2,3,1)) == 2  # one train input and one validation input
    assert calls.count((2,1)) == 2    # both targets as well


def test_optimizer_created_after_model_move(pipeline, monkeypatch):
    original=torch.optim.Adam
    model=TinyModel()
    def checked(*args,**kwargs):
        assert model.moves==['cpu']
        return original(*args,**kwargs)
    monkeypatch.setattr(torch.optim,'Adam',checked)
    pipeline.train.train_model(model,loader(),loader(),{'epochs':1})


@pytest.mark.parametrize("config", [{'epochs':0},{'epochs':1.2},{'patience':0},{'lr':float('nan')},{'lr':0}])
def test_invalid_training_config(pipeline, config):
    with pytest.raises(ValueError):
        pipeline.train.train_model(TinyModel(),loader(),loader(),config)


@pytest.mark.parametrize('empty_role',['train','val'])
def test_empty_loader_fails(pipeline,empty_role):
    args=[loader(0) if empty_role=='train' else loader(),
          loader(0) if empty_role=='val' else loader()]
    with pytest.raises(ValueError):
        pipeline.train.train_model(TinyModel(),*args,{'epochs':1})


def test_loss_denominator_counts_yielded_samples(pipeline):
    class ZeroModel(TinyModel):
        def forward(self,x): return x[:,-1,:]*self.weight*0
    _,history=pipeline.train.train_model(ZeroModel(),loader(5,2,True),loader(5,2,True),{'epochs':1})
    assert history['train_loss']==[1.0]
    assert history['val_loss']==[1.0]


def test_no_silent_shape_broadcasting(pipeline):
    wrong=DataLoader(TensorDataset(torch.ones(2,3,1),torch.ones(2)),batch_size=2)
    with pytest.raises(ValueError):
        pipeline.train.train_model(TinyModel(),wrong,loader(),{'epochs':1})


def test_nonfinite_loss_rejected(pipeline):
    bad=DataLoader(TensorDataset(torch.ones(2,3,1),torch.full((2,1),float('nan'))),batch_size=2)
    with pytest.raises(ValueError):
        pipeline.train.train_model(TinyModel(),bad,loader(),{'epochs':1})


def test_early_stop_restores_best_weights(pipeline):
    class ScriptedValidation(TinyModel):
        def __init__(self): super().__init__(); self.snapshots=[]
        def forward(self,x):
            if not self.training:
                self.snapshots.append(self.weight.detach().clone())
                return self.weight*0 + torch.ones_like(x[:,-1,:])*(1+len(self.snapshots))
            return super().forward(x)
    model=ScriptedValidation()
    result,history=pipeline.train.train_model(model,loader(2),loader(2),{'epochs':5,'patience':1})
    assert len(history['val_loss'])==2
    assert torch.equal(result.weight, model.snapshots[0])
    assert not torch.equal(model.snapshots[0],model.snapshots[1])


def test_predict_moves_model_and_returns_cpu_arrays(pipeline):
    model=TinyModel()
    preds,truth=pipeline.evaluate.predict(model,loader())
    assert model.moves == ['cpu']
    assert preds.shape == truth.shape == (4,1)
    assert not model.training


def test_predict_empty_and_nonfinite(pipeline):
    with pytest.raises(ValueError): pipeline.evaluate.predict(TinyModel(),loader(0))
    bad=DataLoader(TensorDataset(torch.ones(2,3,1),torch.full((2,1),float('nan'))),batch_size=2)
    with pytest.raises(ValueError): pipeline.evaluate.predict(TinyModel(),bad)


def test_metrics_exact(pipeline):
    result=pipeline.evaluate.compute_metrics(np.array([1.,3.]),np.array([2.,2.]))
    assert result['MAE']==1 and result['RMSE']==1 and result['MAPE']==pytest.approx(50)


def test_undefined_metrics_are_not_fabricated(pipeline):
    assert math.isnan(pipeline.evaluate.compute_metrics(np.array([1.,2.]),np.array([0.,2.]))['MAPE'])
    assert math.isnan(pipeline.evaluate.compute_metrics(np.array([1.]),np.array([1.]))['R2'])


@pytest.mark.parametrize('pred,true',[(np.array([]),np.array([])),(np.ones(2),np.ones(3)),
                                    (np.ones((2,2)),np.ones((2,2))),
                                    (np.array([np.inf]),np.ones(1))])
def test_invalid_metric_inputs(pipeline,pred,true):
    with pytest.raises(ValueError): pipeline.evaluate.compute_metrics(pred,true)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='No CUDA device available; not a GPU validation')
def test_cuda_with_cpu_batches(pipeline):
    model,_=pipeline.train.train_model(TinyModel(),loader(),loader(),{'epochs':1},'cuda')
    assert next(model.parameters()).device.type=='cuda'
    preds,_=pipeline.evaluate.predict(model,loader(),'cuda')
    assert np.isfinite(preds).all()


@pytest.mark.parametrize('name',['lstm','gru','attention'])
@pytest.mark.parametrize('horizon',[1,3,7])
def test_real_architectures_end_to_end(pipeline, name, horizon):
    from models.lstm_model import LSTMModel
    from models.gru_model import GRUModel
    from models.attention_model import AttentionModel
    model_type={'lstm':LSTMModel,'gru':GRUModel,'attention':AttentionModel}[name]
    df=pd.DataFrame({'close':100+np.arange(100)*.1},index=pd.date_range('2026-01-01',periods=100))
    frames=pipeline.preprocess.train_val_test_split(df,.6,.2)
    x,y,xv,yv,xt,yt,scaler=pipeline.preprocess.prepare_datasets(*frames,4,horizon)
    tr=DataLoader(TensorDataset(*pipeline.preprocess.numpy_to_torch(x,y)),batch_size=8)
    va=DataLoader(TensorDataset(*pipeline.preprocess.numpy_to_torch(xv,yv)),batch_size=8)
    te=DataLoader(TensorDataset(*pipeline.preprocess.numpy_to_torch(xt,yt)),batch_size=8)
    model,history=pipeline.train.train_model(model_type(1,4,1,1),tr,va,{'epochs':1})
    preds,trues=pipeline.evaluate.predict(model,te)
    metrics=pipeline.evaluate.compute_metrics(scaler.inverse_transform(preds),scaler.inverse_transform(trues))
    assert np.isfinite(list(metrics.values())).all()
    assert len(history['train_loss'])==1
    assert preds.shape==yt.shape
