"?I
BHostIDLE"IDLE1     ??@A     ??@a0?PT???i0?PT????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      b@9      b@A      b@I      b@a???-????i??-?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      Y@9      Y@A      Y@I      Y@a]mOj????i5N??b
???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      Y@9      Y@A      Y@I      Y@a]mOj????i???-?????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1     ?X@9     ?X@A     ?X@I     ?X@a=?S???i?kpH????Unknown
^HostGatherV2"GatherV2(1     ?T@9     ?T@A     ?T@I     ?T@aS(*	[??i"??a ????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     ?I@9     ?I@A     ?I@I     ?I@a?????(??i?X??5???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     ?F@9     ?F@A     ?F@I     ?F@a|G?x~??i?v??????Unknown
u	HostFlushSummaryWriter"FlushSummaryWriter(1      E@9      E@A      E@I      E@a:?u?=???i6N??b
???Unknown?
}
HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      8@9      8@A      8@I      8@a?=?S}?iP??b
E???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      4@9      4@A      4@I      4@a????kpx?ie?
:?u???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      2@I      2@a???-??u?i??f??????Unknown
dHostDataset"Iterator::Model(1      d@9      d@A      1@I      1@ai? ??t?i??g?t????Unknown
oHostSoftmax"sequential/dense_1/Softmax(1      ,@9      ,@A      ,@I      ,@a|G?x~q?iY?Y??????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1      *@9      *@A      *@I      *@a?ͅ?X?o?i'V?q???Unknown
iHostWriteSummary"WriteSummary(1      (@9      (@A      (@I      (@a?=?Sm?i4o.??*???Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      &@9      &@A      &@I      &@a?K???j?i?̦E???Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1      &@9      &@A      &@I      &@a?K???j?i???ۈ`???Unknown
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@a????kph?iW|G?x???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1       @9       @A       @I       @a?f?"?c?i`mOj?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a?f?"?c?ii?"?????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?f?"?c?ir9???????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1       @9       @A       @I       @a?f?"?c?i{???-????Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @a|G?x~a?iBQI????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @a|G?x~a?i	???d????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?=?S]?i?ک????Unknown
YHostPow"Adam/Pow(1      @9      @A      @I      @a????kpX?iT>??F???Unknown
ZHostArgMax"ArgMax(1      @9      @A      @I      @a?f?"?S?iX?q???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?f?"?S?i\?q????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?f?"?S?i`Wۓ?!???Unknown
?HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?f?"?S?id
E%a+???Unknown
~ HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a?=?SM?i?PT?2???Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?=?SM?i??c?
:???Unknown
e"Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?=?SM?i-?r?_A???Unknown?
?#HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?=?SM?ip#?ٴH???Unknown
?$HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?=?SM?i?i??	P???Unknown
?%HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?=?SM?i????^W???Unknown
?&HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      @9      @A      @I      @a?=?SM?i9????^???Unknown
v'HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @a?=?SM?i|<??f???Unknown
?(HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1      @9      @A      @I      @a?=?SM?i???z]m???Unknown
?)HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @a?=?SM?i??g?t???Unknown
?*HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?=?SM?iE?T|???Unknown
t+HostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @a?f?"?C?i?衝?????Unknown
],HostCast"Adam/Cast_1(1       @9       @A       @I       @a?f?"?C?iI?V?ͅ???Unknown
v-HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a?f?"?C?i˛/?????Unknown
v.HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1       @9       @A       @I       @a?f?"?C?iMu?w?????Unknown
[/HostPow"
Adam/Pow_1(1       @9       @A       @I       @a?f?"?C?i?Nu?w????Unknown
V0HostCast"Cast(1       @9       @A       @I       @a?f?"?C?iQ(*	[????Unknown
X1HostCast"Cast_2(1       @9       @A       @I       @a?f?"?C?i??Q>????Unknown
X2HostEqual"Equal(1       @9       @A       @I       @a?f?"?C?iUۓ?!????Unknown
`3HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?f?"?C?i״H?????Unknown
u4HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?f?"?C?iY??+?????Unknown
b5HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?f?"?C?i?g?t˱???Unknown
}6HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?f?"?C?i]Ag??????Unknown
}7HostMul",gradient_tape/sequential/dropout/dropout/Mul(1       @9       @A       @I       @a?f?"?C?i??????Unknown
?8HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?f?"?C?ia??Nu????Unknown
?9HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a?f?"?C?i?ͅ?X????Unknown
?:HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a?f?"?C?ie?:?;????Unknown
q;HostCast"sequential/dropout/dropout/Cast(1       @9       @A       @I       @a?f?"?C?i???(????Unknown
o<HostMul"sequential/dropout/dropout/Mul(1       @9       @A       @I       @a?f?"?C?iiZ?q????Unknown
q=HostMul" sequential/dropout/dropout/Mul_1(1       @9       @A       @I       @a?f?"?C?i?3Y??????Unknown
?>HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1       @9       @A       @I       @a?f?"?C?im?????Unknown
??HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a?f?"?C?i???K?????Unknown
o@HostReadVariableOp"Adam/ReadVariableOp(1      ??9      ??A      ??I      ??a?f?"?3?i?S?????Unknown
tAHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a?f?"?3?iq?w??????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?f?"?3?i2-?8????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?f?"?3?i??,?r????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a?f?"?3?i????????Unknown
XEHostCast"Cast_3(1      ??9      ??A      ??I      ??a?f?"?3?ius?%V????Unknown
XFHostCast"Cast_4(1      ??9      ??A      ??I      ??a?f?"?3?i6?;??????Unknown
TGHostMul"Mul(1      ??9      ??A      ??I      ??a?f?"?3?i?L?n9????Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?f?"?3?i????????Unknown
IHostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1      ??9      ??A      ??I      ??a?f?"?3?iy&K?????Unknown
?JHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?f?"?3?i:??[?????Unknown
?KHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?f?"?3?i?????????Unknown
4LHostIdentity"Identity(i?????????Unknown?
iMHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
JNHostReadVariableOp"div_no_nan_1/ReadVariableOp(i?????????Unknown
LOHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown*?I
sHostDataset"Iterator::Model::ParallelMapV2(1      b@9      b@A      b@I      b@a???????i????????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      Y@9      Y@A      Y@I      Y@a??uO??i???????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      Y@9      Y@A      Y@I      Y@a??uO??i^?1J~????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1     ?X@9     ?X@A     ?X@I     ?X@a?&!??ȹ?i	G?,????Unknown
^HostGatherV2"GatherV2(1     ?T@9     ?T@A     ?T@I     ?T@a36?mֵ?i????$D???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     ?I@9     ?I@A     ?I@I     ?I@a?p?*֪?i????????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     ?F@9     ?F@A     ?F@I     ?F@aε?_魧?i5?fl???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      E@9      E@A      E@I      E@a'!?????iG???????Unknown?
}	HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      8@9      8@A      8@I      8@auJy?	B??i???????Unknown
?
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      4@9      4@A      4@I      4@a?h:?]??i?R??u@???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      2@I      2@a???????i?*?????Unknown
dHostDataset"Iterator::Model(1      d@9      d@A      1@I      1@ah?Kj???i??(?"g???Unknown
oHostSoftmax"sequential/dense_1/Softmax(1      ,@9      ,@A      ,@I      ,@a4,?T?w??iLe{?????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1      *@9      *@A      *@I      *@aT?*?\??i9?#XuJ???Unknown
iHostWriteSummary"WriteSummary(1      (@9      (@A      (@I      (@auJy?	B??ic?!?}????Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      &@9      &@A      &@I      &@a????3'??i?uO???Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1      &@9      &@A      &@I      &@a????3'??i/|??h???Unknown
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@a?h:?]??i?eq??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1       @9       @A       @I       @a???T?ր?i?S?ZC ???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a???T?ր?i
B ?C???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a???T?ր?i&0m??????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1       @9       @A       @I       @a???T?ր?iB??S????Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @a4,?T?w}?i??jC???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @a4,?T?w}?i???2@???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @auJy?	By?i????r???Unknown
YHostPow"Adam/Pow(1      @9      @A      @I      @a?h:?]u?iXfgSϜ???Unknown
ZHostArgMax"ArgMax(1      @9      @A      @I      @a???T??p?if]?|????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a???T??p?itT?*????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???T??p?i?Ke{????Unknown
?HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???T??p?i?Bބ#???Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @auJy?	Bi?iڻ??<???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @auJy?	Bi?i$5?V???Unknown
e!Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @auJy?	Bi?in??Jo???Unknown?
?"HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @auJy?	Bi?i?'?????Unknown
?#HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @auJy?	Bi?i?ϡ???Unknown
?$HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @auJy?	Bi?iL????Unknown
?%HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      @9      @A      @I      @auJy?	Bi?i??$S????Unknown
v&HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @auJy?	Bi?i?.?????Unknown
?'HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1      @9      @A      @I      @auJy?	Bi?i*?
8????Unknown
?(HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @auJy?	Bi?it?	B ???Unknown
?)HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @auJy?	Bi?i?x	L[9???Unknown
t*HostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @a???T??`?iEt^?1J???Unknown
]+HostCast"Adam/Cast_1(1       @9       @A       @I       @a???T??`?i?o??[???Unknown
v,HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a???T??`?iSk`?k???Unknown
v-HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1       @9       @A       @I       @a???T??`?i?f]?|???Unknown
[.HostPow"
Adam/Pow_1(1       @9       @A       @I       @a???T??`?iab?????Unknown
V/HostCast"Cast(1       @9       @A       @I       @a???T??`?i?]tc????Unknown
X0HostCast"Cast_2(1       @9       @A       @I       @a???T??`?ioY\%:????Unknown
X1HostEqual"Equal(1       @9       @A       @I       @a???T??`?i?T??????Unknown
`2HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a???T??`?i}P??????Unknown
u3HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a???T??`?iL[9?????Unknown
b4HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a???T??`?i?G???????Unknown
}5HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a???T??`?iC?k???Unknown
}6HostMul",gradient_tape/sequential/dropout/dropout/Mul(1       @9       @A       @I       @a???T??`?i?>ZMB???Unknown
?7HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a???T??`?i :??%???Unknown
?8HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a???T??`?i?5??5???Unknown
?9HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a???T??`?i.1Ya?F???Unknown
q:HostCast"sequential/dropout/dropout/Cast(1       @9       @A       @I       @a???T??`?i?,??W???Unknown
o;HostMul"sequential/dropout/dropout/Mul(1       @9       @A       @I       @a???T??`?i<(?sh???Unknown
q<HostMul" sequential/dropout/dropout/Mul_1(1       @9       @A       @I       @a???T??`?i?#XuJy???Unknown
?=HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1       @9       @A       @I       @a???T??`?iJ?&!????Unknown
?>HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a???T??`?i???????Unknown
o?HostReadVariableOp"Adam/ReadVariableOp(1      ??9      ??A      ??I      ??a???T??P?i???0c????Unknown
t@HostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a???T??P?iWW?Ϋ???Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a???T??P?i??9????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a???T??P?i??:?????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a???T??P?i??V?????Unknown
XDHostCast"Cast_3(1      ??9      ??A      ??I      ??a???T??P?ic?{????Unknown
XEHostCast"Cast_4(1      ??9      ??A      ??I      ??a???T??P?i&??D?????Unknown
TFHostMul"Mul(1      ??9      ??A      ??I      ??a???T??P?i?V?R????Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???T??P?i?? ??????Unknown
HHostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1      ??9      ??A      ??I      ??a???T??P?io?N)????Unknown
?IHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a???T??P?i2?U??????Unknown
?JHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a???T??P?i?????????Unknown
4KHostIdentity"Identity(i?????????Unknown?
iLHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
JMHostReadVariableOp"div_no_nan_1/ReadVariableOp(i?????????Unknown
LNHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown2CPU