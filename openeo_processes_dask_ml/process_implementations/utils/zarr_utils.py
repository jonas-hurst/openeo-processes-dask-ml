import inspect
from urllib.parse import urlparse

import fsspec
import xarray as xr


def open_zarr_auto(
    path,
    s3_anon=None,
    storage_options=None,
    zarr_format=None,  # None -> auto-detect; or 2 / 3 to force
    **kwargs,
):
    """
    Open a zarr store from a local path, http(s) URL, or s3 URL,
    automatically figuring out storage options, consolidated metadata,
    and the Zarr format version (v2 vs v3).

    Parameters
    ----------
    path : str
        Local path, http(s):// URL, or s3:// URL.
    s3_anon : bool or None
        Force anonymous (True) / credentialed (False) S3 access.
        If None, both are tried automatically.
    storage_options : dict or None
        Extra storage options merged into the auto-detected ones.
    zarr_format : int or None
        2 or 3 to force a format, or None to auto-detect.
    **kwargs :
        Extra keyword args forwarded to xarray.open_zarr.

    Returns
    -------
    xarray.Dataset
    """
    path = str(path)
    scheme = urlparse(path).scheme.lower()
    storage_options = dict(storage_options or {})

    # ---- 1. Build candidate storage_options based on the scheme -------------
    candidates = []
    if scheme in ("http", "https"):
        candidates.append({**storage_options})
    elif scheme == "s3":
        if s3_anon is True:
            candidates.append({"anon": True, **storage_options})
        elif s3_anon is False:
            candidates.append({"anon": False, **storage_options})
        else:
            candidates.append({"anon": False, **storage_options})
            candidates.append({"anon": True, **storage_options})
    elif scheme in ("gs", "gcs"):
        candidates.append({**storage_options})
    else:  # local
        candidates.append({**storage_options})

    # ---- 2. Introspect which kwargs open_zarr actually supports -------------
    # `zarr_version` was renamed to `zarr_format` across versions; only pass
    # what's available to avoid TypeErrors.
    sig_params = set(inspect.signature(xr.open_zarr).parameters)
    fmt_kw = None
    if "zarr_format" in sig_params:
        fmt_kw = "zarr_format"
    elif "zarr_version" in sig_params:
        fmt_kw = "zarr_version"

    # ---- 3. Helpers ---------------------------------------------------------
    def exists(fs, p):
        try:
            return fs.exists(p)
        except Exception:
            return None

    def detect_format(fs, root):
        """Return 2, 3, or None (unknown) by probing store layout."""
        root = root.rstrip("/")
        # Zarr v3: a single zarr.json at the group root
        if exists(fs, root + "/zarr.json") is True:
            return 3
        # Zarr v2: .zgroup / .zmetadata / .zarray
        for marker in (".zmetadata", ".zgroup", ".zarray"):
            if exists(fs, root + "/" + marker) is True:
                return 2
        return None

    def detect_consolidated(fs, root, fmt):
        """Detect whether consolidated metadata is present."""
        root = root.rstrip("/")
        if fmt == 2:
            r = exists(fs, root + "/.zmetadata")
            return r if r is not None else None
        if fmt == 3:
            # v3 consolidated metadata lives inside zarr.json (or a
            # consolidated key). Detection is unreliable, so let xarray
            # decide unless it clearly fails -> return None (try both).
            return None
        return None

    # ---- 4. Try each candidate ---------------------------------------------
    last_err = None
    user_consolidated = kwargs.pop("consolidated", "__unset__")

    for so in candidates:
        try:
            # Build a mapper (remote) or use the plain path (local)
            if scheme in ("http", "https", "s3", "gs", "gcs"):
                mapper = fsspec.get_mapper(path, **so)
                fs, root, so_pass = mapper.fs, path, None
            else:
                mapper = path
                fs, root, so_pass = fsspec.filesystem("file"), path, so or None

            # --- determine format(s) to try ---
            if zarr_format is not None:
                fmt_opts = [zarr_format]
            else:
                detected_fmt = detect_format(fs, root)
                fmt_opts = [detected_fmt] if detected_fmt else [3, 2]

            for fmt in fmt_opts:
                # --- determine consolidated option(s) ---
                if user_consolidated != "__unset__":
                    cons_opts = [user_consolidated]
                else:
                    detected_cons = detect_consolidated(fs, root, fmt)
                    if detected_cons is True:
                        cons_opts = [True]
                    elif detected_cons is False:
                        cons_opts = [False]
                    else:
                        cons_opts = [True, False]

                for consolidated in cons_opts:
                    open_kwargs = dict(kwargs)
                    open_kwargs["consolidated"] = consolidated
                    if fmt_kw and fmt is not None:
                        open_kwargs[fmt_kw] = fmt
                    if so_pass is not None:
                        open_kwargs["storage_options"] = so_pass

                    try:
                        return xr.open_zarr(
                            mapper, **open_kwargs, decode_coords="all", chunks=None
                        )
                    except Exception as e:
                        last_err = e
                        continue

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not open zarr store at {path!r}. Last error: {last_err!r}"
    ) from last_err
