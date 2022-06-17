import os


def _get_user_home():
    return os.path.expanduser('~')


def _get_paddlespcceh_home():
    if 'PPSPEECH_HOME' in os.environ:
        home_path = os.environ['PPSPEECH_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError(
                    'The environment variable PPSPEECH_HOME {} is not a directory.'.
                    format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddlespeech')


def _get_sub_home(directory):
    home = os.path.join(_get_paddlespcceh_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home)
    return home


PPSPEECH_HOME = _get_paddlespcceh_home()
MODEL_HOME = _get_sub_home('models')
CONF_HOME = _get_sub_home('conf')
DATA_HOME = _get_sub_home('datasets')
