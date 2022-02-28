"""Downloads or otherwise fetches pretrained models

Authors:
 * Aku Rouhe 2021
 * Samuele Cornell 2021
"""
import urllib.request
import urllib.error
import pathlib
import logging
import huggingface_hub
from requests.exceptions import HTTPError

logger = logging.getLogger(__name__)


def _missing_ok_unlink(path):
    # missing_ok=True was added to Path.unlink() in Python 3.8
    # This does the same.
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def fetch(
    filename,
    source,
    savedir="./pretrained_model_checkpoints",
    overwrite=False,
    save_filename=None,
    use_auth_token=False,
):
    """Ensures you have a local copy of the file, returns its path

    In case the source is an external location, downloads the file.  In case
    the source is already accessible on the filesystem, creates a symlink in
    the savedir. Thus, the side effects of this function always look similar:
    savedir/save_filename can be used to access the file. And save_filename
    defaults to the filename arg.

    Arguments
    ---------
    filename : str
        Name of the file including extensions.
    source : str
        Where to look for the file. This is interpreted in special ways:
        First, if the source begins with "http://" or "https://", it is
        interpreted as a web address and the file is downloaded.
        Second, if the source is a valid directory path, a symlink is
        created to the file.
        Otherwise, the source is interpreted as a Huggingface model hub ID, and
        the file is downloaded from there.
    savedir : str
        Path where to save downloads/symlinks.
    overwrite : bool
        If True, always overwrite existing savedir/filename file and download
        or recreate the link. If False (as by default), if savedir/filename
        exists, assume it is correct and don't download/relink. Note that
        Huggingface local cache is always used - with overwrite=True we just
        relink from the local cache.
    save_filename : str
        The filename to use for saving this file. Defaults to filename if not
        given.
    use_auth_token : bool (default: False)
        If true Hugginface's auth_token will be used to load private models from the HuggingFace Hub,
        default is False because majority of models are public.
    Returns
    -------
    pathlib.Path
        Path to file on local file system.

    Raises
    ------
    ValueError
        If file is not found
    """
    if save_filename is None:
        save_filename = filename
    savedir = pathlib.Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    sourcefile = f"{source}/{filename}"
    destination = savedir / save_filename
    if destination.exists() and not overwrite:
        MSG = f"Fetch {filename}: Using existing file/symlink in {str(destination)}."
        logger.info(MSG)
        return destination
    if str(source).startswith("http:") or str(source).startswith("https:"):
        # Interpret source as web address.
        MSG = (
            f"Fetch {filename}: Downloading from normal URL {str(sourcefile)}."
        )
        logger.info(MSG)
        # Download
        try:
            urllib.request.urlretrieve(sourcefile, destination)
        except urllib.error.URLError:
            raise ValueError(
                f"Interpreted {source} as web address, but could not download."
            )
    elif pathlib.Path(source).is_dir():
        # Interpret source as local directory path
        # Just symlink
        sourcepath = pathlib.Path(sourcefile).absolute()
        MSG = f"Fetch {filename}: Linking to local file in {str(sourcepath)}."
        logger.info(MSG)
        _missing_ok_unlink(destination)
        destination.symlink_to(sourcepath)
    else:
        # Interpret source as huggingface hub ID
        # Use huggingface hub's fancy cached download.
        MSG = f"Fetch {filename}: Delegating to Huggingface hub, source {str(source)}."
        logger.info(MSG)
        url = huggingface_hub.hf_hub_url(source, filename)
        try:
            fetched_file = huggingface_hub.cached_download(url, use_auth_token)
        except HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError("File not found on HF hub")
            else:
                raise
        # Huggingface hub downloads to etag filename, symlink to the expected one:
        sourcepath = pathlib.Path(fetched_file).absolute()
        _missing_ok_unlink(destination)
        destination.symlink_to(sourcepath)
    return destination
