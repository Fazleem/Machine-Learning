import os

from warnings import filterwarnings

filterwarnings("ignore", category=FutureWarning)

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# import tensorflow as tf
# from tensorflow.python.util import deprecation

# deprecation._PRINT_DEPRECATION_WARNINGS = False
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from keras.models import Sequential
# from keras.layers import Conv2D, BatchNormalization, Dense, Flatten
# from keras.optimizers import RMSprop
# from keras.losses import categorical_crossentropy
# from keras.utils import to_categorical
# import numpy as np

# model = Sequential()
# model.add(BatchNormalization())
# model.add(Conv2D(12, 3, data_format="channels_first", input_shape=(1, 28, 28)))
# model.add(Conv2D(12, 3, data_format="channels_first"))
# model.add(Conv2D(12, 3))
# model.add(Flatten())
# model.add(Dense(10, activation="softmax"))
# model.compile(RMSprop(), categorical_crossentropy, metrics=["accuracy"])
# model.fit(
#     x=np.random.uniform(1, 100, [1000, 1, 28, 28]),
#     y=to_categorical(np.round(np.random.uniform(1, 9, 1000)).ravel()),
# )
# # model.add(Conv2D(64, 3, input_shape=(28, 28, 1)))
try:
    import numpy as np
    from pathlib import Path
    from scipy.io import loadmat
    from warnings import warn

    try:
        input_shape = model.input_shape
    except BaseException as e:
        if isinstance(e, NameError):
            raise NameError(
                """
This code assumes you have saved your keras Sequential model `model`. I.e. you
have a line of code like:

    model = Sequential()

such that you later call model.add(), model.compile(), model.fit() and etc. If
you haven't done this, change your variable name to be "model" to proceed.
"""
            )
        elif isinstance(e, AttributeError):
            try:
                input_shape = model.get_config().get("build_input_shape")
            except BaseException as e2:
                print(e2)
                raise RuntimeError(
                    f"""
You probably haven't called `model.fit(...)` yet. Make sure to only paste the
provided submission code directly after your `model.fit()` call.

Original Error:
{e}
"""
                )

        else:
            raise AttributeError(
                """
Something went wrong trying to load your model and inspect the input shape. Are
you sure you pasted the provided code *after* you were done with all your
model.add() calls, and that you have specified the input shape? Consider asking
the TA for help, or read and understand this code to find the problem"
"""
            )
            print(e)
    if input_shape is None:
        raise ValueError(
            """
You need to specify in your first layer the argument: `input_shape=(28, 28, 1)`
or `input_shape=(1, 28, 28).
"""
        )
    try:
        layers = model.get_config().get("layers")
    except BaseException as e:
        if isinstance(e, AttributeError):
            raise RuntimeError(
                """
Either `model.get_config` could not be found, or `model.get_config() returned a
non-dictionary. You are probably using a very old version of keras or
tensorflow, and should see the TA about running your predictions with your
code.
"""
            )
        else:
            raise RuntimeError(
                """
Couldn't find any layers or couldn't find config. Are you using an old version
of keras or tensorflow without keras?"
"""
            )
    conv_formats = []
    for layer in layers:
        info = layer.get("class_name")
        if info == "Conv2D":
            try:
                conv_formats.append(layer.get("config").get("data_format"))
            except KeyError:
                raise RuntimeError(
                    """
Error trying to find `data_format` for your CNN. Have you specified in your
first convolution-specific layer the argument `data_format` with a value of
either `channels_first` or `channels_last`?
"""
                )
            break  # stop because after first layer, tf ignores format
    unq_conv_formats = np.unique(conv_formats)

    if len(unq_conv_formats) != 1:
        print(f"\n{80*'#'}")
        print("DEREK: WARNING:")
        print(
            f"""
You have more than one data format, or have not specified a data format in your
CNN. Make sure you are only defining the data format (`channels_last` or
`channels_first` once, in the first convolution-specific layer of your network,
and that this format matches your data size. E.g. `channels_first` for an
input_shape of (n, 1, 28, 28) and `channels_last` for an input shape of
(n, 28, 28, 1)).

This is only a warning. If you are able to fit your model and attain good
accuracies, your setup might be fine.

Conv Layer formats: {unq_conv_formats}
"""
        )
        print(f"{80*'#'}\n")
    data_format = unq_conv_formats[0]
    if data_format not in ["channels_first", "channels_last"]:
        raise RuntimeError(
            """
There is probably an error in this validation code. This is probably not your
fault. Contact the TA."
"""
        )
    img_size = input_shape[1:]
    if img_size not in [(28, 28, 1), (1, 28, 28)]:
        warn(
            """
Nonstandard image input shape detected. Perhaps you know what you are doing,
and this is because you have done some kind of preprocessing step that changes
the size of your inputs. If so, then preprocess the provided
`NumberRecognitionTesting.mat` data in the same way, and RUN THE PREDICTIONS
YOURSELF.\n\n But if not, you are probably not passing correctly formatted data
into your CNN.
"""
        )

    if img_size == (28, 28, 1) and data_format == "channels_first":
        raise ValueError(
            """
Your input size and data format are inconsistent. This should not generally be
possible, as Tensorflow should have complained about various shaping issues.
Make sure if your `input_shape` is (28, 28, 1), and your input data (e.g.
X_train) is (n, 28, 28, 1), that the `data_format` is then "channels_last". Fix
this in order to use the prediction-generating code.
"""
        )
    if img_size == (1, 28, 28) and data_format == "channels_last":
        raise ValueError(
            """
Your input size and data format are inconsistent. This should not generally be
possible, as Tensorflow should have complained about various shaping issues.
Make sure if your `input_shape` is (1, 28, 28), and your input data (e.g.
X_train) is (n, 1, 28, 28), that the `data_format` is then "channels_last". Fix
this in order to use the prediction-generating code.
"""
        )

    if img_size not in [(28, 28, 1), (1, 28, 28)]:
        raise RuntimeError(
            """
The provided code for generating predictions is only valid if your model takes
in images of the size (28, 28, 1) or (1, 28, 28).
"""
        )
    # We know X_test will have shape (28, 28, 5000)
    filefolder = Path(__file__).absolute().parent.absolute()
    DATA_PATH = str((filefolder / "NumberRecognitionTesting.mat").absolute())
    X_test = loadmat(DATA_PATH).get("X_test")
    X_test = np.transpose(X_test, [2, 0, 1])  # now shape is (5000, 28, 28)
    X_test = np.array(X_test, dtype=float)  # handle tf2 data errors
    if img_size == (1, 28, 28):  # channels first
        X_test = np.expand_dims(X_test, 1)
    else:  # channels last
        X_test = np.expand_dims(X_test, -1)

    try:
        y_test = np.array(
            [
                4,
                4,
                9,
                5,
                0,
                9,
                0,
                5,
                7,
                3,
                6,
                4,
                0,
                7,
                0,
                1,
                3,
                1,
                3,
                7,
                2,
                1,
                1,
                2,
                3,
                5,
                1,
                2,
                4,
                4,
                3,
                5,
                6,
                0,
                4,
                1,
                9,
                5,
                7,
                8,
                7,
                4,
                3,
                7,
                7,
                3,
                2,
                9,
                7,
                1,
                3,
                6,
                9,
                1,
                7,
                9,
                6,
                5,
                9,
                4,
                8,
                7,
                3,
                9,
                7,
                4,
                2,
                5,
                9,
                0,
                5,
                1,
                0,
                1,
                6,
                7,
                3,
                1,
                7,
                8,
                2,
                2,
                9,
                5,
                1,
                5,
                6,
                3,
                4,
                6,
                4,
                6,
                5,
                4,
                2,
                7,
                1,
                8,
                8,
                1,
                0,
                9,
                0,
                1,
                6,
                4,
                3,
                6,
                1,
                1,
                9,
                5,
                4,
                9,
                0,
                6,
                5,
                2,
                7,
                8,
                4,
                3,
                8,
                9,
                2,
                2,
                4,
                9,
                7,
                2,
                4,
                2,
                5,
                8,
                8,
                5,
                7,
                9,
                1,
                8,
                1,
                0,
                3,
                9,
                1,
                8,
                2,
                1,
                9,
                7,
                5,
                9,
                6,
                1,
                5,
                2,
                0,
                4,
                0,
                0,
                8,
                1,
                2,
                4,
                2,
                3,
                3,
                0,
                3,
                1,
                5,
                5,
                9,
                2,
                0,
                0,
                7,
                1,
                1,
                1,
                3,
                9,
                8,
                5,
                1,
                3,
                8,
                1,
                5,
                5,
                1,
                5,
                7,
                9,
                6,
                2,
                0,
                5,
                6,
                3,
                7,
                4,
                4,
                0,
                6,
                2,
                1,
                9,
                6,
                5,
                4,
                4,
                3,
                3,
                1,
                7,
                7,
                7,
                1,
                4,
                2,
                0,
                4,
                1,
                8,
                4,
                8,
                7,
                0,
                3,
                0,
                6,
                6,
                3,
                3,
                2,
                2,
                6,
                8,
                0,
                6,
                6,
                6,
                2,
                7,
                5,
                8,
                1,
                8,
                1,
                2,
                9,
                1,
                5,
                4,
                0,
                8,
                9,
                9,
                0,
                8,
                0,
                3,
                9,
                5,
                2,
                1,
                3,
                1,
                3,
                5,
                7,
                2,
                5,
                4,
                8,
                9,
                1,
                8,
                1,
                9,
                3,
                4,
                6,
                4,
                1,
                8,
                2,
                5,
                0,
                2,
                7,
                0,
                7,
                4,
                4,
                0,
                9,
                3,
                3,
                3,
                2,
                1,
                7,
                0,
                5,
                3,
                8,
                9,
                8,
                5,
                7,
                6,
                4,
                0,
                2,
                1,
                9,
                0,
                8,
                4,
                6,
                2,
                7,
                2,
                9,
                2,
                3,
                7,
                6,
                7,
                2,
                5,
                8,
                4,
                9,
                8,
                7,
                4,
                4,
                1,
                9,
                5,
                3,
                3,
                3,
                6,
                5,
                1,
                4,
                4,
                1,
                0,
                7,
                7,
                0,
                9,
                5,
                0,
                0,
                8,
                0,
                1,
                7,
                3,
                6,
                7,
                2,
                6,
                9,
                3,
                1,
                4,
                2,
                2,
                0,
                6,
                2,
                7,
                4,
                7,
                4,
                9,
                8,
                4,
                0,
                2,
                4,
                1,
                6,
                4,
                1,
                9,
                4,
                1,
                5,
                3,
                8,
                3,
                1,
                5,
                8,
                4,
                1,
                5,
                2,
                5,
                1,
                2,
                8,
                4,
                3,
                7,
                9,
                3,
                2,
                6,
                1,
                3,
                2,
                1,
                4,
                2,
                4,
                7,
                8,
                1,
                4,
                7,
                1,
                3,
                0,
                7,
                0,
                3,
                7,
                6,
                3,
                2,
                5,
                6,
                0,
                8,
                2,
                8,
                8,
                8,
                3,
                6,
                1,
                2,
                2,
                9,
                0,
                7,
                4,
                6,
                2,
                1,
                4,
                7,
                7,
                8,
                1,
                1,
                3,
                2,
                3,
                2,
                3,
                7,
                4,
                5,
                5,
                8,
                6,
                3,
                7,
                6,
                6,
                2,
                7,
                6,
                9,
                3,
                3,
                7,
                9,
                9,
                5,
                4,
                6,
                2,
                2,
                1,
                1,
                2,
                3,
                8,
                7,
            ],
            dtype=np.uint8,
        )
        scores = model.predict(X_test)
        if scores.shape[1] == 10:
            y_pred = np.array(np.argmax(scores, -1), dtype=np.uint8)  # predicted digit
        else:
            y_pred = scores
        acc = np.mean(y_pred.ravel()[:500] == y_test.ravel()[:500])
    except BaseException as e:
        raise RuntimeError(
            f"""
Something went wrong when trying to generate predictions from X_test with your
model. This is very unusual, you should probably contact the TA to see what
might have gone wrong. Be sure to include this error message, and the original
error below in your e-mail, slack message, or if coming in person.

Original error:
{e}
"""
        )
    mcr = np.round(100 * (1 - acc), 2)
    if acc <= 0.9:
        raise RuntimeError(
            f"""
Your estimated misclassification rate ({mcr}%) is 10% or higher. Even a
fairly straightforward CNN which trains for only one epoch will do much better
than this, so there is almost certainly an error or bug in your code somewhere.

scores.shape: {scores.shape}
y_pred.shape: {y_pred.shape}
"""
        )
    elif acc <= 0.95:
        info = f"""
WARNING:
Your estimated misclassification rate ({mcr}%) is above 5%. While this
does mean your basic model setup and use of this script is essentially
correct, and so predictions will be generated for your model, you really
should be able to easily get a much lower error rate.
            """
    elif 0.95 < acc and acc <= 0.98:
        info = f"""
WARNING:
Your estimated misclassification rate ({mcr}%) is between 3-5%. While this
does mean your basic model setup and use of this script is essentially
correct, and so predictions will be generated for your model, you should
still be able to get lower error rates. Your CNN may not even be performing
better than LDA, QDA, KNN.
"""
    elif 0.98 < acc and acc < 0.99:
        info = f"""
WARNING:
Your estimated misclassification rate ({mcr}%) is between
1-2%. This is getting a lot better, but you should be able to get a lower
error rate still without *too* much effort. Your CNN would probably still
struggle to perform better than KNN with k == 1.
"""
    elif 0.99 <= acc and acc < 0.995:
        info = f"""
GREAT!
Your estimated misclassification rate ({mcr}%) is between 0.5% and 1.0%.
That's really good! It is possible (but very tricky) to go lower though!
Don't forget though: this is just an estimate of your final error rate.
"""
    elif 0.995 <= acc and acc < 0.996:
        info = f"""
WOW!
Your estimated misclassification rate ({mcr}%) is between 0.5% and 0.4%
If this isn't a fluke of the subset of the test set being used for
estimation, that's really impressive! Unless you are doing something *very*
fancy, it is almost impossible to get lower than this.
"""
    else:
        info = f"""
!!!!!
Your estimated misclassification rate ({mcr}%) is 0.4% or lower! If this
isn't a fluke of the subset of the test set being used for estimation,
there's a good chance this will be the lowest rate in the class.
Congrats!
"""
    print(f"\n{80*'#'}")
    np.savetxt(filefolder / "predictions.csv", y_pred, newline=",", fmt="%d")
    print("DEREK:\nSuccessfully generated `predictions.csv` from your model!")
    print(f"{info}")
    print(
        f"""
NOTE:
The misclassification rate above is just an *estimate* based on 500 samples
of the total 5000 your model will be actually tested on. So your actual test
performance could be higher, could be lower. In particular, be very suspicious
if you build a model that seems to perform extremely well on this sample of
the test set, but performs very poorly on your cross-validation of the
`NumberRecognitionBigger.mat` data. Those results probably won't generalize
to the full test set.
        """
    )
    print(f"{80*'#'}\n")
except BaseException as e:
    print(f"\n{80*'#'}")
    print("DEREK:\nThere is a problem with your model setup:")
    if isinstance(e, FileNotFoundError):
        print(
            f"""
{e}

Couldn't find the `NumberRecognitionTesting.mat` file in this directory. If you
have renamed it, ensure it is renamed to `NumberRecognitionTesting.mat`, and if
it is not located in the same folder as this script, ensure that it is moved
here.
"""
        )
    else:
        print(e)
    print(f"{80*'#'}\n")
