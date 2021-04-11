(*****************************************************************************
Copyright 2020 Haiping Chen. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************)

namespace TensorFlowNET.Examples.FSharp

open NumSharp
open type Tensorflow.Binding
open type Tensorflow.KerasApi

module LinearRegressionKeras =

    let prepareData () =
        let train_X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 
                               7.59f, 2.167f, 7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f)
                               .reshape(17, 1)

        let train_Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 
                               2.596f, 2.53f, 1.221f, 2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f)
                               .reshape(17, 1)

        (train_X, train_Y)

    let buildModel (train_X : NDArray) (train_Y : NDArray) =
        let layers = keras.layers

        let inputs = keras.Input(shape = (1).asTensorShape).asTensors
        let outputs = layers.Dense(1).Apply(inputs)
        let model = keras.Model(inputs, outputs)

        model.summary()
        model.compile(loss = keras.losses.MeanSquaredError(),
            optimizer = keras.optimizers.SGD(0.005f),
            metrics = [| "acc" |])
        model.fit(train_X, train_Y, epochs = 100)

        let weights = model.trainable_variables
        print($"weight: {weights.[0].numpy()}, bias: {weights.[1].numpy()}")

    let private run () =
        tf.enable_eager_execution()

        let train_X, train_Y = prepareData()

        buildModel train_X train_Y

        true

    let Example =
        { SciSharpExample.Config = ExampleConfig.Create ("Linear Regression (Keras)", priority = 6)
          Run = run
        }

