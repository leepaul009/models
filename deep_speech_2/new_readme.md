# Deep Speech 2 on PaddlePaddle

## Firstly, plz check the commit number shown as follow
```
commit f9801433701abe54c7cbc442bd509698f44b6f0c
Merge: 950f451 10ee066
Author: Cao Ying <lcy.seso@gmail.com>
Date:   Thu Nov 16 18:06:21 2017 +0800
```

## Secondly, plz apply the changes shown below
### Use manifest.test for inference
In the infer.py, replace the `infer_manifest` with `manifest.test`,
```
add_arg('infer_manifest', str, 'data/librispeech/manifest.test', "Filepath of manifest to infer.")
```

### Calculate the elapsed time of inference
In the `infer.py`, apply the following change
```
ds2_model = DeepSpeech2Model(...)
# record the elapsed time of inference, and we can calculate 10 times average for profiling
time_infer = [] 
result_transcripts = ds2_model.infer_batch(..., iter=10, time_infer=time_infer)
print("batchsize=%d, latency=%g(ms), standard deviation=%g(ms)"
          % (args.num_samples, time_infer[0]*1000.0, time_infer[1]*1000.0))
```
In the `model_utils/model.py`, add the parameter `iters` and `time_infer` to the method `infer_batch` of class `DeepSpeech2Model`
```
def infer_batch(self, ..., iters, time_infer):
```
Then, to calculate the elapesd time, apply the following changes to the method `infer_batch`
```
tms = []
tm_avg = 0
tm_sd = 0
# get average elapsed time of inference
for _ in xrange(iters):
          tm_beg = time.time()
          infer_results = self._inferer.infer(
                    input=infer_data, feeding=feeding_dict)
          tm = time.time() - tm_beg
          tms.append(tm)
          tm_avg = tm_avg + tm
tm_avg = float(tm_avg)/float(iters)
for it in tms:
          tm_sd = tm_sd + (it - tm_avg)*(it - tm_avg)
tm_sd = math.sqrt(tm_sd / iters)
# time_infer[0] record the average elapsed time, and time_infer[1] record the standard deviation
time_infer.append(tm_avg)
time_infer.append(tm_sd)
```

### Additional argments for deepSpeech2
In the `infer.py`, add the argments of `rnn_use_batch` and `use_mkldnn` and let users to decide which rnn mode is used and if mkldnn is used
```
add_arg('rnn_use_batch', bool, True, "rnn_use_batch")
add_arg('use_mkldnn', bool, True, "use_mkldnn")
```
add the argments of `rnn_use_batch` and `use_mkldnn` to `paddle.init`
```
in the paddle.init(use_gpu=args.use_gpu,
                trainer_count=args.trainer_count,
                rnn_use_batch=args.rnn_use_batch,
                use_mkldnn=args.use_mkldnn)
```


### In order to use BatchNorm Fusing, apply the changes to model and network 
In the `model_utils/model.py`, add a parameter `fuse_bn` to the methods `__init__` and `_create_network` of class `DeepSpeech2Model` and make its defalut value as `False`. Then we could create a network with or without BatchNorm Fusing.
```
def __init__(..., fuse_bn=False)
          ...
          self._create_network(..., fuse_bn=fuse_bn)
          ...
def _create_network(..., fuse_bn=False)
          ...
          deep_speech_v2_network(...fuse_bn=fuse_bn)
          ...
```

In the `model_utils/network.py`, we make the `bias_attr` as `True` for following paddle ops,
```
paddle.layer.img_conv
paddle.layer.fc # in bidirectional_gru_bn_layer
paddle.layer.fc # in bidirectional_simple_rnn_bn_layer
```

Add parameter `fuse_bn` to the following functions and make default value with `False`
```
conv_bn_layer(...fuse_bn=fuse_bn)
conv_group(...fuse_bn=fuse_bn)
bidirectional_gru_bn_layer(...fuse_bn=fuse_bn)
bidirectional_simple_rnn_bn_layer(...fuse_bn=fuse_bn)
rnn_group(...fuse_bn=fuse_bn)
deep_speech_v2_network(...fuse_bn=False)
```
In the function `conv_bn_layer`, apply the changes shown below
```
    if fuse_bn == False:
        conv_layer = paddle.layer.img_conv(
            input=input,
            filter_size=filter_size,
            num_channels=num_channels_in,
            num_filters=num_channels_out,
            stride=stride,
            padding=padding,
            act=paddle.activation.Linear(),
            bias_attr=True)
        batch_norm = paddle.layer.batch_norm(input=conv_layer, act=act)
        # reset padding part to 0
        scale_sub_region = paddle.layer.scale_sub_region(
            batch_norm, index_range_data, value=0.0)
        return scale_sub_region
    else:
        conv_layer = paddle.layer.img_conv(
            input=input,
            filter_size=filter_size,
            num_channels=num_channels_in,
            num_filters=num_channels_out,
            stride=stride,
            padding=padding,
            act=act,
            bias_attr=True)
        # reset padding part to 0
        scale_sub_region = paddle.layer.scale_sub_region(
            conv_layer, index_range_data, value=0.0)
        return scale_sub_region
```
In the function `bidirectional_simple_rnn_bn_layer`, apply the changes shown below
```
    if share_weights:
        # input-hidden weights shared between bi-direcitonal rnn.
        if fuse_bn == False:
            input_proj = paddle.layer.fc(
                input=input,
                size=size,
                act=paddle.activation.Linear(),
                bias_attr=True)
            # batch norm is only performed on input-state projection
            input_proj_bn = paddle.layer.batch_norm(
                input=input_proj, act=paddle.activation.Linear())
            # forward and backward in time
            forward_simple_rnn = paddle.layer.recurrent(
                input=input_proj_bn, act=act, reverse=False)
            backward_simple_rnn = paddle.layer.recurrent(
                input=input_proj_bn, act=act, reverse=True)
        else:
            input_proj = paddle.layer.fc(
                input=input,
                size=size,
                act=paddle.activation.Linear(),
                bias_attr=True)
            # forward and backward in time
            forward_simple_rnn = paddle.layer.recurrent(
                input=input_proj, act=act, reverse=False)
            backward_simple_rnn = paddle.layer.recurrent(
                input=input_proj, act=act, reverse=True)

```
In the function `bidirectional_gru_bn_layer`, apply the changes shown below
```
    if fuse_bn == False:
        input_proj_forward = paddle.layer.fc(
            input=input,
            size=size * 3,
            act=paddle.activation.Linear(),
            bias_attr=True)
        input_proj_backward = paddle.layer.fc(
            input=input,
            size=size * 3,
            act=paddle.activation.Linear(),
            bias_attr=True)
        # batch norm is only performed on input-related projections
        input_proj_bn_forward = paddle.layer.batch_norm(
            input=input_proj_forward, act=paddle.activation.Linear())
        input_proj_bn_backward = paddle.layer.batch_norm(
            input=input_proj_backward, act=paddle.activation.Linear())
        # forward and backward in time
        forward_gru = paddle.layer.grumemory(
            input=input_proj_bn_forward, act=act, reverse=False)
        backward_gru = paddle.layer.grumemory(
            input=input_proj_bn_backward, act=act, reverse=True)
        return paddle.layer.concat(input=[forward_gru, backward_gru])
    else:
        input_proj_forward = paddle.layer.fc(
            input=input,
            size=size * 3,
            act=paddle.activation.Linear(),
            bias_attr=True)
        input_proj_backward = paddle.layer.fc(
            input=input,
            size=size * 3,
            act=paddle.activation.Linear(),
            bias_attr=True)
        # forward and backward in time
        forward_gru = paddle.layer.grumemory(
            input=input_proj_forward, act=act, reverse=False)
        backward_gru = paddle.layer.grumemory(
            input=input_proj_backward, act=act, reverse=True)
        return paddle.layer.concat(input=[forward_gru, backward_gru])
```



### Setup the swig for decoder
install and build the dependency
download swig-3.0.12.tar.gz

setup
```
cd decoders/swig
sh setup.sh
```







