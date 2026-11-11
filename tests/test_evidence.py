"""Use disposable Git repositories to check the source-evidence gate."""
import hashlib
import subprocess

import pytest

from scripts.smoke_benchmark import collect_source_evidence, verify_git_sources


def git(root, *arguments):
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, timeout=30
    ).stdout.decode("utf-8").strip()


@pytest.fixture
def checkout(tmp_path):
    paths = ["src/sample.py", "requirements-test.txt", "pytest.ini",
             ".github/workflows/tests.yml", ".gitattributes"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8", newline="\n")
    (tmp_path / ".gitattributes").write_text("* text eol=lf\n", encoding="utf-8", newline="\n")
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Synthetic fixture")
    return tmp_path


def test_source_evidence_has_portable_paths_and_exact_bytes(checkout):
    sha256, blobs = collect_source_evidence(checkout)
    assert len(sha256) == 5
    assert all("\\" not in path for path in sha256)
    for name, digest in sha256.items():
        assert digest == hashlib.sha256((checkout / name).read_bytes()).hexdigest()
    result = verify_git_sources(checkout, blobs)
    assert result == {"status": "VERIFIED", "checkout_sha": git(checkout, "rev-parse", "HEAD"),
                      "source_file_count": 5}


@pytest.mark.parametrize("change", ["modified", "untracked"])
def test_git_evidence_rejects_drift(checkout, change):
    name = "src/sample.py" if change == "modified" else "src/extra.py"
    # Preserve the actual CRLF bytes in the hash: never normalize away drift.
    (checkout / name).write_bytes(b"# changed\r\n")
    _, blobs = collect_source_evidence(checkout)
    with pytest.raises(RuntimeError, match="Source bytes differ"):
        verify_git_sources(checkout, blobs)


def test_git_evidence_rejects_wrong_checkout(checkout):
    _, blobs = collect_source_evidence(checkout)
    with pytest.raises(RuntimeError, match="Checkout identity mismatch"):
        verify_git_sources(checkout, blobs, "0" * 40)
