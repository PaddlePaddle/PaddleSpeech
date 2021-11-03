# PaddleAudio Testing Guide




# Testing
First clone a version of the project by
```
git clone https://github.com/PaddlePaddle/models.git

```
Then install the project in your virtual environment.
```
cd models/PaddleAudio
python setup.py bdist_wheel
pip install -e .[dev]
```
The requirements for testing will be installed along with PaddleAudio.  

Now run
```
pytest test
```

If it goes well, you will see outputs like these:
```
platform linux -- Python 3.7.10, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: ./models/PaddleAudio
plugins: hydra-core-1.0.6
collected 16 items  

test/unit_test/test_backend.py ...........                                                                         [ 68%]
test/unit_test/test_features.py .....                                                                              [100%]

==================================================== warnings summary ====================================================
.
.
.
-- Docs: https://docs.pytest.org/en/stable/warnings.html
============================================ 16 passed, 11 warnings in 6.76s =============================================
```
