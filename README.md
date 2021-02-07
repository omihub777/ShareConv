# ShareConv

Shared Convolution(**SConv**) makes use of the distributive property in convolution.

```shell
Conv: h = w1*x1 + w2*x2 + ... + wc*xc
SConv: h = w*(x1+x2+...+xc) 
             = w*x1 + w*x2 + ... + w*xc
```

SConv shares weights of one kernel across the different channels. We can expect decreasing the number of parameters. This expedites us to create some efficient models, which is crucial in edge-device fields. Here shows some results using SConv.


## TODO


## Done