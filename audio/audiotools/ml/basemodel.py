# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/ml/layers/base.py)
import inspect
import shutil
import tempfile
import typing
from pathlib import Path

import paddle
from paddle import nn


class BaseModel(nn.Layer):
    """This is a class that adds useful save/load functionality to a
    ``paddle.nn.Layer`` object. ``BaseModel`` objects can be saved
    as ``package`` easily, making them super easy to port between
    machines without requiring a ton of dependencies. Files can also be
    saved as just weights, in the standard way.

    >>> class Model(ml.BaseModel):
    >>>     def __init__(self, arg1: float = 1.0):
    >>>         super().__init__()
    >>>         self.arg1 = arg1
    >>>         self.linear = nn.Linear(1, 1)
    >>>
    >>>     def forward(self, x):
    >>>         return self.linear(x)
    >>>
    >>> model1 = Model()
    >>>
    >>> with tempfile.NamedTemporaryFile(suffix=".pth") as f:
    >>>     model1.save(
    >>>         f.name,
    >>>     )
    >>>     model2 = Model.load(f.name)
    >>>     out2 = seed_and_run(model2, x)
    >>>     assert paddle.allclose(out1, out2)
    >>>
    >>>     model1.save(f.name, package=True)
    >>>     model2 = Model.load(f.name)
    >>>     model2.save(f.name, package=False)
    >>>     model3 = Model.load(f.name)
    >>>     out3 = seed_and_run(model3, x)
    >>>
    >>> with tempfile.TemporaryDirectory() as d:
    >>>     model1.save_to_folder(d, {"data": 1.0})
    >>>     Model.load_from_folder(d)

    """

    def save(
            self,
            path: str,
            metadata: dict=None,
            package: bool=False,
            intern: list=[],
            extern: list=[],
            mock: list=[], ):
        """Saves the model, either as a package, or just as
        weights, alongside some specified metadata.

        Parameters
        ----------
        path : str
            Path to save model to.
        metadata : dict, optional
            Any metadata to save alongside the model,
            by default None
        package : bool, optional
            Whether to use ``package`` to save the model in
            a format that is portable, by default True
        intern : list, optional
            List of additional libraries that are internal
            to the model, used with package, by default []
        extern : list, optional
            List of additional libraries that are external to
            the model, used with package, by default []
        mock : list, optional
            List of libraries to mock, used with package,
            by default []

        Returns
        -------
        str
            Path to saved model.
        """
        sig = inspect.signature(self.__class__)
        args = {}

        for key, val in sig.parameters.items():
            arg_val = val.default
            if arg_val is not inspect.Parameter.empty:
                args[key] = arg_val

        # Look up attibutes in self, and if any of them are in args,
        # overwrite them in args.
        for attribute in dir(self):
            if attribute in args:
                args[attribute] = getattr(self, attribute)

        metadata = {} if metadata is None else metadata
        metadata["kwargs"] = args
        if not hasattr(self, "metadata"):
            self.metadata = {}
        self.metadata.update(metadata)

        if not package:
            state_dict = {"state_dict": self.state_dict(), "metadata": metadata}
            paddle.save(state_dict, str(path))
        else:
            self._save_package(path, intern=intern, extern=extern, mock=mock)

        return path

    @property
    def device(self):
        """Gets the device the model is on by looking at the device of
        the first parameter. May not be valid if model is split across
        multiple devices.
        """
        return list(self.parameters())[0].place

    @classmethod
    def load(
            cls,
            location: str,
            *args,
            package_name: str=None,
            strict: bool=False,
            **kwargs, ):
        """Load model from a path. Tries first to load as a package, and if
        that fails, tries to load as weights. The arguments to the class are
        specified inside the model weights file.

        Parameters
        ----------
        location : str
            Path to file.
        package_name : str, optional
            Name of package, by default ``cls.__name__``.
        strict : bool, optional
            Ignore unmatched keys, by default False
        kwargs : dict
            Additional keyword arguments to the model instantiation, if
            not loading from package.

        Returns
        -------
        BaseModel
            A model that inherits from BaseModel.
        """
        try:
            model = cls._load_package(location, package_name=package_name)
        except:
            model_dict = paddle.load(location)
            metadata = model_dict["metadata"]
            metadata["kwargs"].update(kwargs)

            sig = inspect.signature(cls)
            class_keys = list(sig.parameters.keys())
            for k in list(metadata["kwargs"].keys()):
                if k not in class_keys:
                    metadata["kwargs"].pop(k)

            model = cls(*args, **metadata["kwargs"])
            model.set_state_dict(model_dict["state_dict"])
            model.metadata = metadata

        return model

    def _save_package(self, path, intern=[], extern=[], mock=[], **kwargs):
        raise NotImplementedError("Currently Paddle does not support packaging")

    @classmethod
    def _load_package(cls, path, package_name=None):
        raise NotImplementedError("Currently Paddle does not support packaging")

    def save_to_folder(
            self,
            folder: typing.Union[str, Path],
            extra_data: dict=None,
            package: bool=False, ):
        """Dumps a model into a folder, as both a package
        and as weights, as well as anything specified in
        ``extra_data``. ``extra_data`` is a dictionary of other
        pickleable files, with the keys being the paths
        to save them in. The model is saved under a subfolder
        specified by the name of the class (e.g. ``folder/generator/[package, weights].pth``
        if the model name was ``Generator``).

        >>> with tempfile.TemporaryDirectory() as d:
        >>>     extra_data = {
        >>>         "optimizer.pth": optimizer.state_dict()
        >>>     }
        >>>     model.save_to_folder(d, extra_data)
        >>>     Model.load_from_folder(d)

        Parameters
        ----------
        folder : typing.Union[str, Path]
            _description_
        extra_data : dict, optional
            _description_, by default None

        Returns
        -------
        str
            Path to folder
        """
        extra_data = {} if extra_data is None else extra_data
        model_name = type(self).__name__.lower()
        target_base = Path(f"{folder}/{model_name}/")
        target_base.mkdir(exist_ok=True, parents=True)

        if package:
            package_path = target_base / f"package.pth"
            self.save(package_path)

        weights_path = target_base / f"weights.pth"
        self.save(weights_path, package=False)

        for path, obj in extra_data.items():
            paddle.save(obj, str(target_base / path))

        return target_base

    @classmethod
    def load_from_folder(
            cls,
            folder: typing.Union[str, Path],
            package: bool=False,
            strict: bool=False,
            **kwargs, ):
        """Loads the model from a folder generated by
        :py:func:`audiotools.ml.layers.base.BaseModel.save_to_folder`.
        Like that function, this one looks for a subfolder that has
        the name of the class (e.g. ``folder/generator/[package, weights].pth`` if the
        model name was ``Generator``).

        Parameters
        ----------
        folder : typing.Union[str, Path]
            _description_
        package : bool, optional
            Whether to use ``package`` to load the model,
            loading the model from ``package.pth``.
        strict : bool, optional
            Ignore unmatched keys, by default False

        Returns
        -------
        tuple
            tuple of model and extra data as saved by
            :py:func:`audiotools.ml.layers.base.BaseModel.save_to_folder`.
        """
        folder = Path(folder) / cls.__name__.lower()
        model_pth = "package.pth" if package else "weights.pth"
        model_pth = folder / model_pth

        model = cls.load(str(model_pth))
        extra_data = {}
        excluded = ["package.pth", "weights.pth"]
        files = [
            x for x in folder.glob("*")
            if x.is_file() and x.name not in excluded
        ]
        for f in files:
            extra_data[f.name] = paddle.load(str(f), **kwargs)

        return model, extra_data
