��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02unknown8�
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: **
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:**
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:***
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:***
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:**
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

: **
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:**
dtype0
�
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:***&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:***
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:**
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

: **
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:**
dtype0
�
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:***&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:***
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:**
dtype0

NoOpNoOp
�>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�>
value�>B�> B�>
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
e
5slice_indices
6regularization_losses
7trainable_variables
8	variables
9	keras_api
R
:regularization_losses
;trainable_variables
<	variables
=	keras_api
R
>regularization_losses
?trainable_variables
@	variables
A	keras_api
R
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
R
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
�
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem�m�m�m�#m�$m�)m�*m�/m�0m�v�v�v�v�#v�$v�)v�*v�/v�0v�
 
F
0
1
2
3
#4
$5
)6
*7
/8
09
F
0
1
2
3
#4
$5
)6
*7
/8
09
�
Onon_trainable_variables
regularization_losses
Player_regularization_losses
Qmetrics
trainable_variables
	variables

Rlayers
 
 
 
 
�
Snon_trainable_variables
regularization_losses
Tlayer_regularization_losses
Umetrics
trainable_variables
	variables

Vlayers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Wnon_trainable_variables
regularization_losses
Xlayer_regularization_losses
Ymetrics
trainable_variables
	variables

Zlayers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
[non_trainable_variables
regularization_losses
\layer_regularization_losses
]metrics
 trainable_variables
!	variables

^layers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
�
_non_trainable_variables
%regularization_losses
`layer_regularization_losses
ametrics
&trainable_variables
'	variables

blayers
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
�
cnon_trainable_variables
+regularization_losses
dlayer_regularization_losses
emetrics
,trainable_variables
-	variables

flayers
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
�
gnon_trainable_variables
1regularization_losses
hlayer_regularization_losses
imetrics
2trainable_variables
3	variables

jlayers

k0
l1
m2
n3
 
 
 
�
onon_trainable_variables
6regularization_losses
player_regularization_losses
qmetrics
7trainable_variables
8	variables

rlayers
 
 
 
�
snon_trainable_variables
:regularization_losses
tlayer_regularization_losses
umetrics
;trainable_variables
<	variables

vlayers
 
 
 
�
wnon_trainable_variables
>regularization_losses
xlayer_regularization_losses
ymetrics
?trainable_variables
@	variables

zlayers
 
 
 
�
{non_trainable_variables
Bregularization_losses
|layer_regularization_losses
}metrics
Ctrainable_variables
D	variables

~layers
 
 
 
�
non_trainable_variables
Fregularization_losses
 �layer_regularization_losses
�metrics
Gtrainable_variables
H	variables
�layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_input_pitchPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_pitchdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������
:���������
*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_450954
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_451692
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/v*/
Tin(
&2$*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_451809٠
�
^
B__inference_lambda_layer_call_and_return_conditional_losses_451234

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:���������2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�	
�
C__inference_dense_3_layer_call_and_return_conditional_losses_450510

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: **
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������*2

Softplus�
IdentityIdentitySoftplus:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������*2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
Z
>__inference_p0_layer_call_and_return_conditional_losses_450721

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_sliceg
SigmoidSigmoidstrided_slice:output:0*
T0*'
_output_shapes
:���������
2	
SigmoidS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x`
mulMulmul/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
mulL
NegNegmul:z:0*
T0*'
_output_shapes
:���������
2
NegW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2	
mul_1/yb
mul_1MulNeg:y:0mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
mul_1S
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
Pow/x^
PowPowPow/x:output:0	mul_1:z:0*
T0*'
_output_shapes
:���������
2
PowS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0Pow:z:0*
T0*'
_output_shapes
:���������
2
sub
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_slice_1m
	Sigmoid_1Sigmoidstrided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
	Sigmoid_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2	
mul_2/xh
mul_2Mulmul_2/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
mul_2Q
imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
imag^
ComplexComplexsub:z:0imag:output:0*'
_output_shapes
:���������
2	
ComplexQ
realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
reald
	Complex_1Complexreal:output:0	mul_2:z:0*'
_output_shapes
:���������
2
	Complex_1T
ExpExpComplex_1:out:0*
T0*'
_output_shapes
:���������
2
Exp_
mul_3MulComplex:out:0Exp:y:0*
T0*'
_output_shapes
:���������
2
mul_3]
IdentityIdentity	mul_3:z:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
Z
>__inference_Rd_layer_call_and_return_conditional_losses_450740

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
A
%__inference_Gain_layer_call_fn_451394

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_Gain_layer_call_and_return_conditional_losses_4507712
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_450532

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:***
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������*2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������*::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
G__inference_slice_layer_layer_call_and_return_conditional_losses_451361	
input
identity

identity_1

identity_2

identity_3{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
strided_slice_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    *   2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinputstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
strided_slice_3j
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:���������2

Identityp

Identity_1Identitystrided_slice_1:output:0*
T0*'
_output_shapes
:���������2

Identity_1p

Identity_2Identitystrided_slice_2:output:0*
T0*'
_output_shapes
:���������2

Identity_2p

Identity_3Identitystrided_slice_3:output:0*
T0*'
_output_shapes
:���������2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*&
_input_shapes
:���������*:% !

_user_specified_nameinput
�
�
&__inference_model_layer_call_fn_451228

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������
:���������
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4509052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_451331

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:***
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������*2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������*::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
^
B__inference_lambda_layer_call_and_return_conditional_losses_451240

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:���������2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_450402
input_pitch.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource0
,model_dense_3_matmul_readvariableop_resource1
-model_dense_3_biasadd_readvariableop_resource0
,model_dense_4_matmul_readvariableop_resource1
-model_dense_4_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�$model/dense_4/BiasAdd/ReadVariableOp�#model/dense_4/MatMul/ReadVariableOpu
model/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
model/lambda/truediv/y�
model/lambda/truedivRealDivinput_pitchmodel/lambda/truediv/y:output:0*
T0*'
_output_shapes
:���������2
model/lambda/truediv�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMulmodel/lambda/truediv:z:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense/BiasAdd�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense_1/MatMul/ReadVariableOp�
model/dense_1/MatMulMatMulmodel/dense/BiasAdd:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_1/MatMul�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_1/BiasAdd�
model/dense_1/SoftplusSoftplusmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/dense_1/Softplus�
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_2/MatMul/ReadVariableOp�
model/dense_2/MatMulMatMul$model/dense_1/Softplus:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense_2/MatMul�
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense_2/BiasAdd�
model/dense_2/SoftplusSoftplusmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model/dense_2/Softplus�
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes

: **
dtype02%
#model/dense_3/MatMul/ReadVariableOp�
model/dense_3/MatMulMatMul$model/dense_2/Softplus:activations:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
model/dense_3/MatMul�
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:**
dtype02&
$model/dense_3/BiasAdd/ReadVariableOp�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
model/dense_3/BiasAdd�
model/dense_3/SoftplusSoftplusmodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������*2
model/dense_3/Softplus�
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

:***
dtype02%
#model/dense_4/MatMul/ReadVariableOp�
model/dense_4/MatMulMatMul$model/dense_3/Softplus:activations:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
model/dense_4/MatMul�
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:**
dtype02&
$model/dense_4/BiasAdd/ReadVariableOp�
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
model/dense_4/BiasAdd�
%model/slice_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%model/slice_layer/strided_slice/stack�
'model/slice_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'model/slice_layer/strided_slice/stack_1�
'model/slice_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'model/slice_layer/strided_slice/stack_2�
model/slice_layer/strided_sliceStridedSlicemodel/dense_4/BiasAdd:output:0.model/slice_layer/strided_slice/stack:output:00model/slice_layer/strided_slice/stack_1:output:00model/slice_layer/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2!
model/slice_layer/strided_slice�
'model/slice_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model/slice_layer/strided_slice_1/stack�
)model/slice_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model/slice_layer/strided_slice_1/stack_1�
)model/slice_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)model/slice_layer/strided_slice_1/stack_2�
!model/slice_layer/strided_slice_1StridedSlicemodel/dense_4/BiasAdd:output:00model/slice_layer/strided_slice_1/stack:output:02model/slice_layer/strided_slice_1/stack_1:output:02model/slice_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2#
!model/slice_layer/strided_slice_1�
'model/slice_layer/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model/slice_layer/strided_slice_2/stack�
)model/slice_layer/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model/slice_layer/strided_slice_2/stack_1�
)model/slice_layer/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)model/slice_layer/strided_slice_2/stack_2�
!model/slice_layer/strided_slice_2StridedSlicemodel/dense_4/BiasAdd:output:00model/slice_layer/strided_slice_2/stack:output:02model/slice_layer/strided_slice_2/stack_1:output:02model/slice_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2#
!model/slice_layer/strided_slice_2�
'model/slice_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model/slice_layer/strided_slice_3/stack�
)model/slice_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    *   2+
)model/slice_layer/strided_slice_3/stack_1�
)model/slice_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)model/slice_layer/strided_slice_3/stack_2�
!model/slice_layer/strided_slice_3StridedSlicemodel/dense_4/BiasAdd:output:00model/slice_layer/strided_slice_3/stack:output:02model/slice_layer/strided_slice_3/stack_1:output:02model/slice_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2#
!model/slice_layer/strided_slice_3�
model/z0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
model/z0/strided_slice/stack�
model/z0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
model/z0/strided_slice/stack_1�
model/z0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
model/z0/strided_slice/stack_2�
model/z0/strided_sliceStridedSlice*model/slice_layer/strided_slice_3:output:0%model/z0/strided_slice/stack:output:0'model/z0/strided_slice/stack_1:output:0'model/z0/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
model/z0/strided_slice�
model/z0/SigmoidSigmoidmodel/z0/strided_slice:output:0*
T0*'
_output_shapes
:���������
2
model/z0/Sigmoide
model/z0/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
model/z0/mul/x�
model/z0/mulMulmodel/z0/mul/x:output:0model/z0/Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
model/z0/mulg
model/z0/NegNegmodel/z0/mul:z:0*
T0*'
_output_shapes
:���������
2
model/z0/Negi
model/z0/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2
model/z0/mul_1/y�
model/z0/mul_1Mulmodel/z0/Neg:y:0model/z0/mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
model/z0/mul_1e
model/z0/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
model/z0/Pow/x�
model/z0/PowPowmodel/z0/Pow/x:output:0model/z0/mul_1:z:0*
T0*'
_output_shapes
:���������
2
model/z0/Powe
model/z0/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
model/z0/sub/x�
model/z0/subSubmodel/z0/sub/x:output:0model/z0/Pow:z:0*
T0*'
_output_shapes
:���������
2
model/z0/sub�
model/z0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
model/z0/strided_slice_1/stack�
 model/z0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 model/z0/strided_slice_1/stack_1�
 model/z0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model/z0/strided_slice_1/stack_2�
model/z0/strided_slice_1StridedSlice*model/slice_layer/strided_slice_3:output:0'model/z0/strided_slice_1/stack:output:0)model/z0/strided_slice_1/stack_1:output:0)model/z0/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
model/z0/strided_slice_1�
model/z0/Sigmoid_1Sigmoid!model/z0/strided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
model/z0/Sigmoid_1i
model/z0/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2
model/z0/mul_2/x�
model/z0/mul_2Mulmodel/z0/mul_2/x:output:0model/z0/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
model/z0/mul_2c
model/z0/imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/z0/imag�
model/z0/ComplexComplexmodel/z0/sub:z:0model/z0/imag:output:0*'
_output_shapes
:���������
2
model/z0/Complexc
model/z0/realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/z0/real�
model/z0/Complex_1Complexmodel/z0/real:output:0model/z0/mul_2:z:0*'
_output_shapes
:���������
2
model/z0/Complex_1o
model/z0/ExpExpmodel/z0/Complex_1:out:0*
T0*'
_output_shapes
:���������
2
model/z0/Exp�
model/z0/mul_3Mulmodel/z0/Complex:out:0model/z0/Exp:y:0*
T0*'
_output_shapes
:���������
2
model/z0/mul_3�
model/p0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
model/p0/strided_slice/stack�
model/p0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
model/p0/strided_slice/stack_1�
model/p0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
model/p0/strided_slice/stack_2�
model/p0/strided_sliceStridedSlice*model/slice_layer/strided_slice_2:output:0%model/p0/strided_slice/stack:output:0'model/p0/strided_slice/stack_1:output:0'model/p0/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
model/p0/strided_slice�
model/p0/SigmoidSigmoidmodel/p0/strided_slice:output:0*
T0*'
_output_shapes
:���������
2
model/p0/Sigmoide
model/p0/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
model/p0/mul/x�
model/p0/mulMulmodel/p0/mul/x:output:0model/p0/Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
model/p0/mulg
model/p0/NegNegmodel/p0/mul:z:0*
T0*'
_output_shapes
:���������
2
model/p0/Negi
model/p0/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2
model/p0/mul_1/y�
model/p0/mul_1Mulmodel/p0/Neg:y:0model/p0/mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
model/p0/mul_1e
model/p0/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
model/p0/Pow/x�
model/p0/PowPowmodel/p0/Pow/x:output:0model/p0/mul_1:z:0*
T0*'
_output_shapes
:���������
2
model/p0/Powe
model/p0/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
model/p0/sub/x�
model/p0/subSubmodel/p0/sub/x:output:0model/p0/Pow:z:0*
T0*'
_output_shapes
:���������
2
model/p0/sub�
model/p0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
model/p0/strided_slice_1/stack�
 model/p0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 model/p0/strided_slice_1/stack_1�
 model/p0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model/p0/strided_slice_1/stack_2�
model/p0/strided_slice_1StridedSlice*model/slice_layer/strided_slice_2:output:0'model/p0/strided_slice_1/stack:output:0)model/p0/strided_slice_1/stack_1:output:0)model/p0/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
model/p0/strided_slice_1�
model/p0/Sigmoid_1Sigmoid!model/p0/strided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
model/p0/Sigmoid_1i
model/p0/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2
model/p0/mul_2/x�
model/p0/mul_2Mulmodel/p0/mul_2/x:output:0model/p0/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
model/p0/mul_2c
model/p0/imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/p0/imag�
model/p0/ComplexComplexmodel/p0/sub:z:0model/p0/imag:output:0*'
_output_shapes
:���������
2
model/p0/Complexc
model/p0/realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/p0/real�
model/p0/Complex_1Complexmodel/p0/real:output:0model/p0/mul_2:z:0*'
_output_shapes
:���������
2
model/p0/Complex_1o
model/p0/ExpExpmodel/p0/Complex_1:out:0*
T0*'
_output_shapes
:���������
2
model/p0/Exp�
model/p0/mul_3Mulmodel/p0/Complex:out:0model/p0/Exp:y:0*
T0*'
_output_shapes
:���������
2
model/p0/mul_3e
model/Rd/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
model/Rd/mul/x�
model/Rd/mulMulmodel/Rd/mul/x:output:0*model/slice_layer/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2
model/Rd/muli
model/Gain/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
model/Gain/mul/x�
model/Gain/mulMulmodel/Gain/mul/x:output:0(model/slice_layer/strided_slice:output:0*
T0*'
_output_shapes
:���������2
model/Gain/mul�
IdentityIdentitymodel/Gain/mul:z:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identitymodel/Rd/mul:z:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identitymodel/p0/mul_3:z:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identitymodel/z0/mul_3:z:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp:+ '
%
_user_specified_nameinput_pitch
�5
�
A__inference_model_layer_call_and_return_conditional_losses_450905

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity

identity_1

identity_2

identity_3��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_4504182
lambda/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4504412
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4504642!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4504872!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4505102!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4505322!
dense_4/StatefulPartitionedCall�
slice_layer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������:���������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_slice_layer_layer_call_and_return_conditional_losses_4505672
slice_layer/PartitionedCall�
z0/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_z0_layer_call_and_return_conditional_losses_4506462
z0/PartitionedCall�
p0/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_p0_layer_call_and_return_conditional_losses_4507212
p0/PartitionedCall�
Rd/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_Rd_layer_call_and_return_conditional_losses_4507462
Rd/PartitionedCall�
Gain/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_Gain_layer_call_and_return_conditional_losses_4507712
Gain/PartitionedCall�
IdentityIdentityGain/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1IdentityRd/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identityp0/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identityz0/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_451186

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOpi
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
lambda/truediv/y�
lambda/truedivRealDivinputslambda/truediv/y:output:0*
T0*'
_output_shapes
:���������2
lambda/truediv�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMullambda/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAdd�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAdd|
dense_1/SoftplusSoftplusdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Softplus�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Softplus:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_2/BiasAdd|
dense_2/SoftplusSoftplusdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_2/Softplus�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: **
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldense_2/Softplus:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:**
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
dense_3/BiasAdd|
dense_3/SoftplusSoftplusdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������*2
dense_3/Softplus�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:***
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldense_3/Softplus:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:**
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
dense_4/BiasAdd�
slice_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
slice_layer/strided_slice/stack�
!slice_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!slice_layer/strided_slice/stack_1�
!slice_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!slice_layer/strided_slice/stack_2�
slice_layer/strided_sliceStridedSlicedense_4/BiasAdd:output:0(slice_layer/strided_slice/stack:output:0*slice_layer/strided_slice/stack_1:output:0*slice_layer/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
slice_layer/strided_slice�
!slice_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!slice_layer/strided_slice_1/stack�
#slice_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#slice_layer/strided_slice_1/stack_1�
#slice_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#slice_layer/strided_slice_1/stack_2�
slice_layer/strided_slice_1StridedSlicedense_4/BiasAdd:output:0*slice_layer/strided_slice_1/stack:output:0,slice_layer/strided_slice_1/stack_1:output:0,slice_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
slice_layer/strided_slice_1�
!slice_layer/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!slice_layer/strided_slice_2/stack�
#slice_layer/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#slice_layer/strided_slice_2/stack_1�
#slice_layer/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#slice_layer/strided_slice_2/stack_2�
slice_layer/strided_slice_2StridedSlicedense_4/BiasAdd:output:0*slice_layer/strided_slice_2/stack:output:0,slice_layer/strided_slice_2/stack_1:output:0,slice_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
slice_layer/strided_slice_2�
!slice_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!slice_layer/strided_slice_3/stack�
#slice_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    *   2%
#slice_layer/strided_slice_3/stack_1�
#slice_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#slice_layer/strided_slice_3/stack_2�
slice_layer/strided_slice_3StridedSlicedense_4/BiasAdd:output:0*slice_layer/strided_slice_3/stack:output:0,slice_layer/strided_slice_3/stack_1:output:0,slice_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
slice_layer/strided_slice_3�
z0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
z0/strided_slice/stack�
z0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
z0/strided_slice/stack_1�
z0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
z0/strided_slice/stack_2�
z0/strided_sliceStridedSlice$slice_layer/strided_slice_3:output:0z0/strided_slice/stack:output:0!z0/strided_slice/stack_1:output:0!z0/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
z0/strided_slicep

z0/SigmoidSigmoidz0/strided_slice:output:0*
T0*'
_output_shapes
:���������
2

z0/SigmoidY
z0/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2

z0/mul/xl
z0/mulMulz0/mul/x:output:0z0/Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
z0/mulU
z0/NegNeg
z0/mul:z:0*
T0*'
_output_shapes
:���������
2
z0/Neg]

z0/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2

z0/mul_1/yn
z0/mul_1Mul
z0/Neg:y:0z0/mul_1/y:output:0*
T0*'
_output_shapes
:���������
2

z0/mul_1Y
z0/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

z0/Pow/xj
z0/PowPowz0/Pow/x:output:0z0/mul_1:z:0*
T0*'
_output_shapes
:���������
2
z0/PowY
z0/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

z0/sub/xh
z0/subSubz0/sub/x:output:0
z0/Pow:z:0*
T0*'
_output_shapes
:���������
2
z0/sub�
z0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
z0/strided_slice_1/stack�
z0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
z0/strided_slice_1/stack_1�
z0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
z0/strided_slice_1/stack_2�
z0/strided_slice_1StridedSlice$slice_layer/strided_slice_3:output:0!z0/strided_slice_1/stack:output:0#z0/strided_slice_1/stack_1:output:0#z0/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
z0/strided_slice_1v
z0/Sigmoid_1Sigmoidz0/strided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
z0/Sigmoid_1]

z0/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2

z0/mul_2/xt
z0/mul_2Mulz0/mul_2/x:output:0z0/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2

z0/mul_2W
z0/imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
z0/imagj

z0/ComplexComplex
z0/sub:z:0z0/imag:output:0*'
_output_shapes
:���������
2

z0/ComplexW
z0/realConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
z0/realp
z0/Complex_1Complexz0/real:output:0z0/mul_2:z:0*'
_output_shapes
:���������
2
z0/Complex_1]
z0/ExpExpz0/Complex_1:out:0*
T0*'
_output_shapes
:���������
2
z0/Expk
z0/mul_3Mulz0/Complex:out:0
z0/Exp:y:0*
T0*'
_output_shapes
:���������
2

z0/mul_3�
p0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
p0/strided_slice/stack�
p0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
p0/strided_slice/stack_1�
p0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
p0/strided_slice/stack_2�
p0/strided_sliceStridedSlice$slice_layer/strided_slice_2:output:0p0/strided_slice/stack:output:0!p0/strided_slice/stack_1:output:0!p0/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
p0/strided_slicep

p0/SigmoidSigmoidp0/strided_slice:output:0*
T0*'
_output_shapes
:���������
2

p0/SigmoidY
p0/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2

p0/mul/xl
p0/mulMulp0/mul/x:output:0p0/Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
p0/mulU
p0/NegNeg
p0/mul:z:0*
T0*'
_output_shapes
:���������
2
p0/Neg]

p0/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2

p0/mul_1/yn
p0/mul_1Mul
p0/Neg:y:0p0/mul_1/y:output:0*
T0*'
_output_shapes
:���������
2

p0/mul_1Y
p0/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

p0/Pow/xj
p0/PowPowp0/Pow/x:output:0p0/mul_1:z:0*
T0*'
_output_shapes
:���������
2
p0/PowY
p0/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

p0/sub/xh
p0/subSubp0/sub/x:output:0
p0/Pow:z:0*
T0*'
_output_shapes
:���������
2
p0/sub�
p0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
p0/strided_slice_1/stack�
p0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
p0/strided_slice_1/stack_1�
p0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
p0/strided_slice_1/stack_2�
p0/strided_slice_1StridedSlice$slice_layer/strided_slice_2:output:0!p0/strided_slice_1/stack:output:0#p0/strided_slice_1/stack_1:output:0#p0/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
p0/strided_slice_1v
p0/Sigmoid_1Sigmoidp0/strided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
p0/Sigmoid_1]

p0/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2

p0/mul_2/xt
p0/mul_2Mulp0/mul_2/x:output:0p0/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2

p0/mul_2W
p0/imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
p0/imagj

p0/ComplexComplex
p0/sub:z:0p0/imag:output:0*'
_output_shapes
:���������
2

p0/ComplexW
p0/realConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
p0/realp
p0/Complex_1Complexp0/real:output:0p0/mul_2:z:0*'
_output_shapes
:���������
2
p0/Complex_1]
p0/ExpExpp0/Complex_1:out:0*
T0*'
_output_shapes
:���������
2
p0/Expk
p0/mul_3Mulp0/Complex:out:0
p0/Exp:y:0*
T0*'
_output_shapes
:���������
2

p0/mul_3Y
Rd/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Rd/mul/x�
Rd/mulMulRd/mul/x:output:0$slice_layer/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2
Rd/mul]

Gain/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2

Gain/mul/x�
Gain/mulMulGain/mul/x:output:0"slice_layer/strided_slice:output:0*
T0*'
_output_shapes
:���������2

Gain/mul�
IdentityIdentityGain/mul:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity
Rd/mul:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identityp0/mul_3:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identityz0/mul_3:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_dense_4_layer_call_fn_451338

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4505322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������*2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������*::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_450924
input_pitch"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_pitchstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������
:���������
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4509052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_pitch
�
�
$__inference_signature_wrapper_450954
input_pitch"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_pitchstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������
:���������
*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_4504022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_pitch
�
Z
>__inference_p0_layer_call_and_return_conditional_losses_451447

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_sliceg
SigmoidSigmoidstrided_slice:output:0*
T0*'
_output_shapes
:���������
2	
SigmoidS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x`
mulMulmul/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
mulL
NegNegmul:z:0*
T0*'
_output_shapes
:���������
2
NegW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2	
mul_1/yb
mul_1MulNeg:y:0mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
mul_1S
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
Pow/x^
PowPowPow/x:output:0	mul_1:z:0*
T0*'
_output_shapes
:���������
2
PowS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0Pow:z:0*
T0*'
_output_shapes
:���������
2
sub
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_slice_1m
	Sigmoid_1Sigmoidstrided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
	Sigmoid_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2	
mul_2/xh
mul_2Mulmul_2/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
mul_2Q
imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
imag^
ComplexComplexsub:z:0imag:output:0*'
_output_shapes
:���������
2	
ComplexQ
realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
reald
	Complex_1Complexreal:output:0	mul_2:z:0*'
_output_shapes
:���������
2
	Complex_1T
ExpExpComplex_1:out:0*
T0*'
_output_shapes
:���������
2
Exp_
mul_3MulComplex:out:0Exp:y:0*
T0*'
_output_shapes
:���������
2
mul_3]
IdentityIdentity	mul_3:z:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_451070

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOpi
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
lambda/truediv/y�
lambda/truedivRealDivinputslambda/truediv/y:output:0*
T0*'
_output_shapes
:���������2
lambda/truediv�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMullambda/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAdd�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAdd|
dense_1/SoftplusSoftplusdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Softplus�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Softplus:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_2/BiasAdd|
dense_2/SoftplusSoftplusdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_2/Softplus�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: **
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldense_2/Softplus:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:**
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
dense_3/BiasAdd|
dense_3/SoftplusSoftplusdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������*2
dense_3/Softplus�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:***
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldense_3/Softplus:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:**
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
dense_4/BiasAdd�
slice_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
slice_layer/strided_slice/stack�
!slice_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!slice_layer/strided_slice/stack_1�
!slice_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!slice_layer/strided_slice/stack_2�
slice_layer/strided_sliceStridedSlicedense_4/BiasAdd:output:0(slice_layer/strided_slice/stack:output:0*slice_layer/strided_slice/stack_1:output:0*slice_layer/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
slice_layer/strided_slice�
!slice_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!slice_layer/strided_slice_1/stack�
#slice_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#slice_layer/strided_slice_1/stack_1�
#slice_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#slice_layer/strided_slice_1/stack_2�
slice_layer/strided_slice_1StridedSlicedense_4/BiasAdd:output:0*slice_layer/strided_slice_1/stack:output:0,slice_layer/strided_slice_1/stack_1:output:0,slice_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
slice_layer/strided_slice_1�
!slice_layer/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!slice_layer/strided_slice_2/stack�
#slice_layer/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#slice_layer/strided_slice_2/stack_1�
#slice_layer/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#slice_layer/strided_slice_2/stack_2�
slice_layer/strided_slice_2StridedSlicedense_4/BiasAdd:output:0*slice_layer/strided_slice_2/stack:output:0,slice_layer/strided_slice_2/stack_1:output:0,slice_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
slice_layer/strided_slice_2�
!slice_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!slice_layer/strided_slice_3/stack�
#slice_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    *   2%
#slice_layer/strided_slice_3/stack_1�
#slice_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#slice_layer/strided_slice_3/stack_2�
slice_layer/strided_slice_3StridedSlicedense_4/BiasAdd:output:0*slice_layer/strided_slice_3/stack:output:0,slice_layer/strided_slice_3/stack_1:output:0,slice_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
slice_layer/strided_slice_3�
z0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
z0/strided_slice/stack�
z0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
z0/strided_slice/stack_1�
z0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
z0/strided_slice/stack_2�
z0/strided_sliceStridedSlice$slice_layer/strided_slice_3:output:0z0/strided_slice/stack:output:0!z0/strided_slice/stack_1:output:0!z0/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
z0/strided_slicep

z0/SigmoidSigmoidz0/strided_slice:output:0*
T0*'
_output_shapes
:���������
2

z0/SigmoidY
z0/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2

z0/mul/xl
z0/mulMulz0/mul/x:output:0z0/Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
z0/mulU
z0/NegNeg
z0/mul:z:0*
T0*'
_output_shapes
:���������
2
z0/Neg]

z0/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2

z0/mul_1/yn
z0/mul_1Mul
z0/Neg:y:0z0/mul_1/y:output:0*
T0*'
_output_shapes
:���������
2

z0/mul_1Y
z0/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

z0/Pow/xj
z0/PowPowz0/Pow/x:output:0z0/mul_1:z:0*
T0*'
_output_shapes
:���������
2
z0/PowY
z0/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

z0/sub/xh
z0/subSubz0/sub/x:output:0
z0/Pow:z:0*
T0*'
_output_shapes
:���������
2
z0/sub�
z0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
z0/strided_slice_1/stack�
z0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
z0/strided_slice_1/stack_1�
z0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
z0/strided_slice_1/stack_2�
z0/strided_slice_1StridedSlice$slice_layer/strided_slice_3:output:0!z0/strided_slice_1/stack:output:0#z0/strided_slice_1/stack_1:output:0#z0/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
z0/strided_slice_1v
z0/Sigmoid_1Sigmoidz0/strided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
z0/Sigmoid_1]

z0/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2

z0/mul_2/xt
z0/mul_2Mulz0/mul_2/x:output:0z0/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2

z0/mul_2W
z0/imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
z0/imagj

z0/ComplexComplex
z0/sub:z:0z0/imag:output:0*'
_output_shapes
:���������
2

z0/ComplexW
z0/realConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
z0/realp
z0/Complex_1Complexz0/real:output:0z0/mul_2:z:0*'
_output_shapes
:���������
2
z0/Complex_1]
z0/ExpExpz0/Complex_1:out:0*
T0*'
_output_shapes
:���������
2
z0/Expk
z0/mul_3Mulz0/Complex:out:0
z0/Exp:y:0*
T0*'
_output_shapes
:���������
2

z0/mul_3�
p0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
p0/strided_slice/stack�
p0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
p0/strided_slice/stack_1�
p0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
p0/strided_slice/stack_2�
p0/strided_sliceStridedSlice$slice_layer/strided_slice_2:output:0p0/strided_slice/stack:output:0!p0/strided_slice/stack_1:output:0!p0/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
p0/strided_slicep

p0/SigmoidSigmoidp0/strided_slice:output:0*
T0*'
_output_shapes
:���������
2

p0/SigmoidY
p0/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2

p0/mul/xl
p0/mulMulp0/mul/x:output:0p0/Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
p0/mulU
p0/NegNeg
p0/mul:z:0*
T0*'
_output_shapes
:���������
2
p0/Neg]

p0/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2

p0/mul_1/yn
p0/mul_1Mul
p0/Neg:y:0p0/mul_1/y:output:0*
T0*'
_output_shapes
:���������
2

p0/mul_1Y
p0/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

p0/Pow/xj
p0/PowPowp0/Pow/x:output:0p0/mul_1:z:0*
T0*'
_output_shapes
:���������
2
p0/PowY
p0/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

p0/sub/xh
p0/subSubp0/sub/x:output:0
p0/Pow:z:0*
T0*'
_output_shapes
:���������
2
p0/sub�
p0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
p0/strided_slice_1/stack�
p0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
p0/strided_slice_1/stack_1�
p0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
p0/strided_slice_1/stack_2�
p0/strided_slice_1StridedSlice$slice_layer/strided_slice_2:output:0!p0/strided_slice_1/stack:output:0#p0/strided_slice_1/stack_1:output:0#p0/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
p0/strided_slice_1v
p0/Sigmoid_1Sigmoidp0/strided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
p0/Sigmoid_1]

p0/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2

p0/mul_2/xt
p0/mul_2Mulp0/mul_2/x:output:0p0/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2

p0/mul_2W
p0/imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
p0/imagj

p0/ComplexComplex
p0/sub:z:0p0/imag:output:0*'
_output_shapes
:���������
2

p0/ComplexW
p0/realConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
p0/realp
p0/Complex_1Complexp0/real:output:0p0/mul_2:z:0*'
_output_shapes
:���������
2
p0/Complex_1]
p0/ExpExpp0/Complex_1:out:0*
T0*'
_output_shapes
:���������
2
p0/Expk
p0/mul_3Mulp0/Complex:out:0
p0/Exp:y:0*
T0*'
_output_shapes
:���������
2

p0/mul_3Y
Rd/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Rd/mul/x�
Rd/mulMulRd/mul/x:output:0$slice_layer/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2
Rd/mul]

Gain/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2

Gain/mul/x�
Gain/mulMulGain/mul/x:output:0"slice_layer/strided_slice:output:0*
T0*'
_output_shapes
:���������2

Gain/mul�
IdentityIdentityGain/mul:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity
Rd/mul:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identityp0/mul_3:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identityz0/mul_3:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_450464

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Softplus�
IdentityIdentitySoftplus:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�F
�
__inference__traced_save_451692
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1077b68044ae46d7a5e3345af8e350c7/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::: : : *:*:**:*: : : : : ::::: : : *:*:**:*::::: : : *:*:**:*: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
^
B__inference_lambda_layer_call_and_return_conditional_losses_450412

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:���������2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
Z
>__inference_p0_layer_call_and_return_conditional_losses_451478

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_sliceg
SigmoidSigmoidstrided_slice:output:0*
T0*'
_output_shapes
:���������
2	
SigmoidS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x`
mulMulmul/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
mulL
NegNegmul:z:0*
T0*'
_output_shapes
:���������
2
NegW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2	
mul_1/yb
mul_1MulNeg:y:0mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
mul_1S
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
Pow/x^
PowPowPow/x:output:0	mul_1:z:0*
T0*'
_output_shapes
:���������
2
PowS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0Pow:z:0*
T0*'
_output_shapes
:���������
2
sub
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_slice_1m
	Sigmoid_1Sigmoidstrided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
	Sigmoid_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2	
mul_2/xh
mul_2Mulmul_2/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
mul_2Q
imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
imag^
ComplexComplexsub:z:0imag:output:0*'
_output_shapes
:���������
2	
ComplexQ
realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
reald
	Complex_1Complexreal:output:0	mul_2:z:0*'
_output_shapes
:���������
2
	Complex_1T
ExpExpComplex_1:out:0*
T0*'
_output_shapes
:���������
2
Exp_
mul_3MulComplex:out:0Exp:y:0*
T0*'
_output_shapes
:���������
2
mul_3]
IdentityIdentity	mul_3:z:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
&__inference_dense_layer_call_fn_451267

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4504412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
C
'__inference_lambda_layer_call_fn_451245

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_4504122
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
\
@__inference_Gain_layer_call_and_return_conditional_losses_450771

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
C
'__inference_lambda_layer_call_fn_451250

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_4504182
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
?
#__inference_z0_layer_call_fn_451555

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_z0_layer_call_and_return_conditional_losses_4506152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_451296

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2

Softplus�
IdentityIdentitySoftplus:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
?
#__inference_p0_layer_call_fn_451488

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_p0_layer_call_and_return_conditional_losses_4507212
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
G__inference_slice_layer_layer_call_and_return_conditional_losses_450567	
input
identity

identity_1

identity_2

identity_3{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
strided_slice_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    *   2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinputstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask2
strided_slice_3j
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:���������2

Identityp

Identity_1Identitystrided_slice_1:output:0*
T0*'
_output_shapes
:���������2

Identity_1p

Identity_2Identitystrided_slice_2:output:0*
T0*'
_output_shapes
:���������2

Identity_2p

Identity_3Identitystrided_slice_3:output:0*
T0*'
_output_shapes
:���������2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*&
_input_shapes
:���������*:% !

_user_specified_nameinput
�
Z
>__inference_p0_layer_call_and_return_conditional_losses_450690

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_sliceg
SigmoidSigmoidstrided_slice:output:0*
T0*'
_output_shapes
:���������
2	
SigmoidS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x`
mulMulmul/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
mulL
NegNegmul:z:0*
T0*'
_output_shapes
:���������
2
NegW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2	
mul_1/yb
mul_1MulNeg:y:0mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
mul_1S
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
Pow/x^
PowPowPow/x:output:0	mul_1:z:0*
T0*'
_output_shapes
:���������
2
PowS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0Pow:z:0*
T0*'
_output_shapes
:���������
2
sub
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_slice_1m
	Sigmoid_1Sigmoidstrided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
	Sigmoid_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2	
mul_2/xh
mul_2Mulmul_2/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
mul_2Q
imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
imag^
ComplexComplexsub:z:0imag:output:0*'
_output_shapes
:���������
2	
ComplexQ
realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
reald
	Complex_1Complexreal:output:0	mul_2:z:0*'
_output_shapes
:���������
2
	Complex_1T
ExpExpComplex_1:out:0*
T0*'
_output_shapes
:���������
2
Exp_
mul_3MulComplex:out:0Exp:y:0*
T0*'
_output_shapes
:���������
2
mul_3]
IdentityIdentity	mul_3:z:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
?
#__inference_Rd_layer_call_fn_451411

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_Rd_layer_call_and_return_conditional_losses_4507402
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_451285

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4504642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
?
#__inference_z0_layer_call_fn_451560

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_z0_layer_call_and_return_conditional_losses_4506462
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_451278

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Softplus�
IdentityIdentitySoftplus:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
A
%__inference_Gain_layer_call_fn_451389

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_Gain_layer_call_and_return_conditional_losses_4507652
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
Z
>__inference_Rd_layer_call_and_return_conditional_losses_451406

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
Z
>__inference_z0_layer_call_and_return_conditional_losses_451550

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_sliceg
SigmoidSigmoidstrided_slice:output:0*
T0*'
_output_shapes
:���������
2	
SigmoidS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x`
mulMulmul/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
mulL
NegNegmul:z:0*
T0*'
_output_shapes
:���������
2
NegW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2	
mul_1/yb
mul_1MulNeg:y:0mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
mul_1S
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
Pow/x^
PowPowPow/x:output:0	mul_1:z:0*
T0*'
_output_shapes
:���������
2
PowS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0Pow:z:0*
T0*'
_output_shapes
:���������
2
sub
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_slice_1m
	Sigmoid_1Sigmoidstrided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
	Sigmoid_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2	
mul_2/xh
mul_2Mulmul_2/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
mul_2Q
imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
imag^
ComplexComplexsub:z:0imag:output:0*'
_output_shapes
:���������
2	
ComplexQ
realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
reald
	Complex_1Complexreal:output:0	mul_2:z:0*'
_output_shapes
:���������
2
	Complex_1T
ExpExpComplex_1:out:0*
T0*'
_output_shapes
:���������
2
Exp_
mul_3MulComplex:out:0Exp:y:0*
T0*'
_output_shapes
:���������
2
mul_3]
IdentityIdentity	mul_3:z:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_450441

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_450872
input_pitch"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_pitchstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������
:���������
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4508532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_pitch
�	
�
C__inference_dense_3_layer_call_and_return_conditional_losses_451314

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: **
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������*2

Softplus�
IdentityIdentitySoftplus:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������*2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
?
#__inference_p0_layer_call_fn_451483

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_p0_layer_call_and_return_conditional_losses_4506902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_451260

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
Z
>__inference_z0_layer_call_and_return_conditional_losses_450646

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_sliceg
SigmoidSigmoidstrided_slice:output:0*
T0*'
_output_shapes
:���������
2	
SigmoidS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x`
mulMulmul/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
mulL
NegNegmul:z:0*
T0*'
_output_shapes
:���������
2
NegW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2	
mul_1/yb
mul_1MulNeg:y:0mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
mul_1S
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
Pow/x^
PowPowPow/x:output:0	mul_1:z:0*
T0*'
_output_shapes
:���������
2
PowS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0Pow:z:0*
T0*'
_output_shapes
:���������
2
sub
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_slice_1m
	Sigmoid_1Sigmoidstrided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
	Sigmoid_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2	
mul_2/xh
mul_2Mulmul_2/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
mul_2Q
imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
imag^
ComplexComplexsub:z:0imag:output:0*'
_output_shapes
:���������
2	
ComplexQ
realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
reald
	Complex_1Complexreal:output:0	mul_2:z:0*'
_output_shapes
:���������
2
	Complex_1T
ExpExpComplex_1:out:0*
T0*'
_output_shapes
:���������
2
Exp_
mul_3MulComplex:out:0Exp:y:0*
T0*'
_output_shapes
:���������
2
mul_3]
IdentityIdentity	mul_3:z:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�5
�
A__inference_model_layer_call_and_return_conditional_losses_450788
input_pitch(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity

identity_1

identity_2

identity_3��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
lambda/PartitionedCallPartitionedCallinput_pitch*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_4504122
lambda/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4504412
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4504642!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4504872!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4505102!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4505322!
dense_4/StatefulPartitionedCall�
slice_layer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������:���������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_slice_layer_layer_call_and_return_conditional_losses_4505672
slice_layer/PartitionedCall�
z0/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_z0_layer_call_and_return_conditional_losses_4506152
z0/PartitionedCall�
p0/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_p0_layer_call_and_return_conditional_losses_4506902
p0/PartitionedCall�
Rd/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_Rd_layer_call_and_return_conditional_losses_4507402
Rd/PartitionedCall�
Gain/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_Gain_layer_call_and_return_conditional_losses_4507652
Gain/PartitionedCall�
IdentityIdentityGain/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1IdentityRd/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identityp0/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identityz0/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:+ '
%
_user_specified_nameinput_pitch
�5
�
A__inference_model_layer_call_and_return_conditional_losses_450819
input_pitch(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity

identity_1

identity_2

identity_3��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
lambda/PartitionedCallPartitionedCallinput_pitch*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_4504182
lambda/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4504412
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4504642!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4504872!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4505102!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4505322!
dense_4/StatefulPartitionedCall�
slice_layer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������:���������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_slice_layer_layer_call_and_return_conditional_losses_4505672
slice_layer/PartitionedCall�
z0/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_z0_layer_call_and_return_conditional_losses_4506462
z0/PartitionedCall�
p0/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_p0_layer_call_and_return_conditional_losses_4507212
p0/PartitionedCall�
Rd/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_Rd_layer_call_and_return_conditional_losses_4507462
Rd/PartitionedCall�
Gain/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_Gain_layer_call_and_return_conditional_losses_4507712
Gain/PartitionedCall�
IdentityIdentityGain/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1IdentityRd/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identityp0/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identityz0/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:+ '
%
_user_specified_nameinput_pitch
�
Z
>__inference_z0_layer_call_and_return_conditional_losses_450615

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_sliceg
SigmoidSigmoidstrided_slice:output:0*
T0*'
_output_shapes
:���������
2	
SigmoidS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x`
mulMulmul/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
mulL
NegNegmul:z:0*
T0*'
_output_shapes
:���������
2
NegW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2	
mul_1/yb
mul_1MulNeg:y:0mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
mul_1S
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
Pow/x^
PowPowPow/x:output:0	mul_1:z:0*
T0*'
_output_shapes
:���������
2
PowS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0Pow:z:0*
T0*'
_output_shapes
:���������
2
sub
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_slice_1m
	Sigmoid_1Sigmoidstrided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
	Sigmoid_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2	
mul_2/xh
mul_2Mulmul_2/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
mul_2Q
imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
imag^
ComplexComplexsub:z:0imag:output:0*'
_output_shapes
:���������
2	
ComplexQ
realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
reald
	Complex_1Complexreal:output:0	mul_2:z:0*'
_output_shapes
:���������
2
	Complex_1T
ExpExpComplex_1:out:0*
T0*'
_output_shapes
:���������
2
Exp_
mul_3MulComplex:out:0Exp:y:0*
T0*'
_output_shapes
:���������
2
mul_3]
IdentityIdentity	mul_3:z:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
\
@__inference_Gain_layer_call_and_return_conditional_losses_450765

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_451207

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������
:���������
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4508532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
\
@__inference_Gain_layer_call_and_return_conditional_losses_451384

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_450487

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2

Softplus�
IdentityIdentitySoftplus:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
Z
>__inference_Rd_layer_call_and_return_conditional_losses_450746

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
?
#__inference_Rd_layer_call_fn_451416

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_Rd_layer_call_and_return_conditional_losses_4507462
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_451809
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias%
!assignvariableop_8_dense_4_kernel#
assignvariableop_9_dense_4_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate+
'assignvariableop_15_adam_dense_kernel_m)
%assignvariableop_16_adam_dense_bias_m-
)assignvariableop_17_adam_dense_1_kernel_m+
'assignvariableop_18_adam_dense_1_bias_m-
)assignvariableop_19_adam_dense_2_kernel_m+
'assignvariableop_20_adam_dense_2_bias_m-
)assignvariableop_21_adam_dense_3_kernel_m+
'assignvariableop_22_adam_dense_3_bias_m-
)assignvariableop_23_adam_dense_4_kernel_m+
'assignvariableop_24_adam_dense_4_bias_m+
'assignvariableop_25_adam_dense_kernel_v)
%assignvariableop_26_adam_dense_bias_v-
)assignvariableop_27_adam_dense_1_kernel_v+
'assignvariableop_28_adam_dense_1_bias_v-
)assignvariableop_29_adam_dense_2_kernel_v+
'assignvariableop_30_adam_dense_2_bias_v-
)assignvariableop_31_adam_dense_3_kernel_v+
'assignvariableop_32_adam_dense_3_bias_v-
)assignvariableop_33_adam_dense_4_kernel_v+
'assignvariableop_34_adam_dense_4_bias_v
identity_36��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_1_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_1_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_2_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_2_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_3_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_3_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_4_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_4_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_3_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_3_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_4_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_4_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35�
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�
�
(__inference_dense_3_layer_call_fn_451321

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4505102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������*2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_451303

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4504872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
w
,__inference_slice_layer_layer_call_fn_451372	
input
identity

identity_1

identity_2

identity_3�
PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������:���������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_slice_layer_layer_call_and_return_conditional_losses_4505672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identityp

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������2

Identity_1p

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:���������2

Identity_2p

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:���������2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*&
_input_shapes
:���������*:% !

_user_specified_nameinput
�5
�
A__inference_model_layer_call_and_return_conditional_losses_450853

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity

identity_1

identity_2

identity_3��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_4504122
lambda/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4504412
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4504642!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4504872!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4505102!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4505322!
dense_4/StatefulPartitionedCall�
slice_layer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*`
_output_shapesN
L:���������:���������:���������:���������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_slice_layer_layer_call_and_return_conditional_losses_4505672
slice_layer/PartitionedCall�
z0/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_z0_layer_call_and_return_conditional_losses_4506152
z0/PartitionedCall�
p0/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_p0_layer_call_and_return_conditional_losses_4506902
p0/PartitionedCall�
Rd/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_Rd_layer_call_and_return_conditional_losses_4507402
Rd/PartitionedCall�
Gain/PartitionedCallPartitionedCall$slice_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_Gain_layer_call_and_return_conditional_losses_4507652
Gain/PartitionedCall�
IdentityIdentityGain/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1IdentityRd/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identityp0/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_2�

Identity_3Identityz0/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*N
_input_shapes=
;:���������::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
Z
>__inference_z0_layer_call_and_return_conditional_losses_451519

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_sliceg
SigmoidSigmoidstrided_slice:output:0*
T0*'
_output_shapes
:���������
2	
SigmoidS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x`
mulMulmul/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
2
mulL
NegNegmul:z:0*
T0*'
_output_shapes
:���������
2
NegW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=2	
mul_1/yb
mul_1MulNeg:y:0mul_1/y:output:0*
T0*'
_output_shapes
:���������
2
mul_1S
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
Pow/x^
PowPowPow/x:output:0	mul_1:z:0*
T0*'
_output_shapes
:���������
2
PowS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0Pow:z:0*
T0*'
_output_shapes
:���������
2
sub
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask2
strided_slice_1m
	Sigmoid_1Sigmoidstrided_slice_1:output:0*
T0*'
_output_shapes
:���������
2
	Sigmoid_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *�I@2	
mul_2/xh
mul_2Mulmul_2/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:���������
2
mul_2Q
imagConst*
_output_shapes
: *
dtype0*
valueB
 *    2
imag^
ComplexComplexsub:z:0imag:output:0*'
_output_shapes
:���������
2	
ComplexQ
realConst*
_output_shapes
: *
dtype0*
valueB
 *    2
reald
	Complex_1Complexreal:output:0	mul_2:z:0*'
_output_shapes
:���������
2
	Complex_1T
ExpExpComplex_1:out:0*
T0*'
_output_shapes
:���������
2
Exp_
mul_3MulComplex:out:0Exp:y:0*
T0*'
_output_shapes
:���������
2
mul_3]
IdentityIdentity	mul_3:z:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
^
B__inference_lambda_layer_call_and_return_conditional_losses_450418

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:���������2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
Z
>__inference_Rd_layer_call_and_return_conditional_losses_451400

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
\
@__inference_Gain_layer_call_and_return_conditional_losses_451378

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �B2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_pitch4
serving_default_input_pitch:0���������8
Gain0
StatefulPartitionedCall:0���������6
Rd0
StatefulPartitionedCall:1���������6
p00
StatefulPartitionedCall:2���������
6
z00
StatefulPartitionedCall:3���������
tensorflow/serving/predict:��
�	
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"�
_tf_keras_model�{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model"}, "training_config": {"loss": ["<lambda>", "<lambda>", "<lambda>", "<lambda>"], "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_pitch", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 1], "config": {"batch_input_shape": [null, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_pitch"}}
�
regularization_losses
trainable_variables
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": ["4wEAAAAAAAAAAQAAAAIAAABDAAAAcwgAAAB8AGQBGwBTACkCTmcAAAAAAABZQKkAKQHaAXhyAQAA\nAHIBAAAA+h48aXB5dGhvbi1pbnB1dC0zLWI4YjNhNjk5NGJkMz7aCDxsYW1iZGE+CwAAAPMAAAAA\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}}
�

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "softplus", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
�

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "softplus", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}}
�

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 42, "activation": "softplus", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
�

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 42, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 42}}}}
�
5slice_indices
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "SliceLayer", "name": "slice_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�
:regularization_losses
;trainable_variables
<	variables
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "Gain", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Gain", "trainable": true, "dtype": "float32", "function": ["4wEAAAAAAAAAAQAAAAIAAABDAAAAcwgAAABkAXwAFABTACkCTmcAAAAAAABZQKkAKQHaAXhyAQAA\nAHIBAAAA+h48aXB5dGhvbi1pbnB1dC0zLWI4YjNhNjk5NGJkMz7aCDxsYW1iZGE+FgAAAPMAAAAA\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
�
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "Rd", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Rd", "trainable": true, "dtype": "float32", "function": ["4wEAAAAAAAAAAQAAAAIAAABDAAAAcwgAAABkAXwAFABTACkCTmcAAAAAAADwP6kAKQHaAXhyAQAA\nAHIBAAAA+h48aXB5dGhvbi1pbnB1dC0zLWI4YjNhNjk5NGJkMz7aCDxsYW1iZGE+FwAAAPMAAAAA\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
�
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "p0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "p0", "trainable": true, "dtype": "float32", "function": ["4wEAAAAAAAAAAQAAAAIAAABDAAAAcwgAAAB0AHwAgwFTACkBTikB2gp0b19jb21wbGV4KQHaAXip\nAHIDAAAA+h48aXB5dGhvbi1pbnB1dC0zLWI4YjNhNjk5NGJkMz7aCDxsYW1iZGE+GAAAAPMAAAAA\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
�
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "z0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "z0", "trainable": true, "dtype": "float32", "function": ["4wEAAAAAAAAAAQAAAAIAAABDAAAAcwgAAAB0AHwAgwFTACkBTikB2gp0b19jb21wbGV4KQHaAXip\nAHIDAAAA+h48aXB5dGhvbi1pbnB1dC0zLWI4YjNhNjk5NGJkMz7aCDxsYW1iZGE+GQAAAPMAAAAA\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
�
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem�m�m�m�#m�$m�)m�*m�/m�0m�v�v�v�v�#v�$v�)v�*v�/v�0v�"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
#4
$5
)6
*7
/8
09"
trackable_list_wrapper
f
0
1
2
3
#4
$5
)6
*7
/8
09"
trackable_list_wrapper
�
Onon_trainable_variables
regularization_losses
Player_regularization_losses
Qmetrics
trainable_variables
	variables

Rlayers
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Snon_trainable_variables
regularization_losses
Tlayer_regularization_losses
Umetrics
trainable_variables
	variables

Vlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Wnon_trainable_variables
regularization_losses
Xlayer_regularization_losses
Ymetrics
trainable_variables
	variables

Zlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
[non_trainable_variables
regularization_losses
\layer_regularization_losses
]metrics
 trainable_variables
!	variables

^layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_2/kernel
: 2dense_2/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
�
_non_trainable_variables
%regularization_losses
`layer_regularization_losses
ametrics
&trainable_variables
'	variables

blayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 : *2dense_3/kernel
:*2dense_3/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�
cnon_trainable_variables
+regularization_losses
dlayer_regularization_losses
emetrics
,trainable_variables
-	variables

flayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :**2dense_4/kernel
:*2dense_4/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
�
gnon_trainable_variables
1regularization_losses
hlayer_regularization_losses
imetrics
2trainable_variables
3	variables

jlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
<
k0
l1
m2
n3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
onon_trainable_variables
6regularization_losses
player_regularization_losses
qmetrics
7trainable_variables
8	variables

rlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
snon_trainable_variables
:regularization_losses
tlayer_regularization_losses
umetrics
;trainable_variables
<	variables

vlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables
>regularization_losses
xlayer_regularization_losses
ymetrics
?trainable_variables
@	variables

zlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
{non_trainable_variables
Bregularization_losses
|layer_regularization_losses
}metrics
Ctrainable_variables
D	variables

~layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables
Fregularization_losses
 �layer_regularization_losses
�metrics
Gtrainable_variables
H	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
%:# 2Adam/dense_2/kernel/m
: 2Adam/dense_2/bias/m
%:# *2Adam/dense_3/kernel/m
:*2Adam/dense_3/bias/m
%:#**2Adam/dense_4/kernel/m
:*2Adam/dense_4/bias/m
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
%:# 2Adam/dense_2/kernel/v
: 2Adam/dense_2/bias/v
%:# *2Adam/dense_3/kernel/v
:*2Adam/dense_3/bias/v
%:#**2Adam/dense_4/kernel/v
:*2Adam/dense_4/bias/v
�2�
A__inference_model_layer_call_and_return_conditional_losses_451186
A__inference_model_layer_call_and_return_conditional_losses_450819
A__inference_model_layer_call_and_return_conditional_losses_451070
A__inference_model_layer_call_and_return_conditional_losses_450788�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_model_layer_call_fn_450872
&__inference_model_layer_call_fn_451228
&__inference_model_layer_call_fn_451207
&__inference_model_layer_call_fn_450924�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_450402�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_pitch���������
�2�
B__inference_lambda_layer_call_and_return_conditional_losses_451240
B__inference_lambda_layer_call_and_return_conditional_losses_451234�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_lambda_layer_call_fn_451250
'__inference_lambda_layer_call_fn_451245�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_dense_layer_call_and_return_conditional_losses_451260�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_layer_call_fn_451267�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_451278�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_451285�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_451296�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_2_layer_call_fn_451303�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_3_layer_call_and_return_conditional_losses_451314�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_3_layer_call_fn_451321�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_4_layer_call_and_return_conditional_losses_451331�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_4_layer_call_fn_451338�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_slice_layer_layer_call_and_return_conditional_losses_451361�
���
FullArgSpec
args�
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_slice_layer_layer_call_fn_451372�
���
FullArgSpec
args�
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_Gain_layer_call_and_return_conditional_losses_451384
@__inference_Gain_layer_call_and_return_conditional_losses_451378�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_Gain_layer_call_fn_451389
%__inference_Gain_layer_call_fn_451394�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
>__inference_Rd_layer_call_and_return_conditional_losses_451400
>__inference_Rd_layer_call_and_return_conditional_losses_451406�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference_Rd_layer_call_fn_451411
#__inference_Rd_layer_call_fn_451416�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
>__inference_p0_layer_call_and_return_conditional_losses_451447
>__inference_p0_layer_call_and_return_conditional_losses_451478�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference_p0_layer_call_fn_451488
#__inference_p0_layer_call_fn_451483�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
>__inference_z0_layer_call_and_return_conditional_losses_451550
>__inference_z0_layer_call_and_return_conditional_losses_451519�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference_z0_layer_call_fn_451560
#__inference_z0_layer_call_fn_451555�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
7B5
$__inference_signature_wrapper_450954input_pitch�
@__inference_Gain_layer_call_and_return_conditional_losses_451378`7�4
-�*
 �
inputs���������

 
p
� "%�"
�
0���������
� �
@__inference_Gain_layer_call_and_return_conditional_losses_451384`7�4
-�*
 �
inputs���������

 
p 
� "%�"
�
0���������
� |
%__inference_Gain_layer_call_fn_451389S7�4
-�*
 �
inputs���������

 
p
� "����������|
%__inference_Gain_layer_call_fn_451394S7�4
-�*
 �
inputs���������

 
p 
� "�����������
>__inference_Rd_layer_call_and_return_conditional_losses_451400`7�4
-�*
 �
inputs���������

 
p
� "%�"
�
0���������
� �
>__inference_Rd_layer_call_and_return_conditional_losses_451406`7�4
-�*
 �
inputs���������

 
p 
� "%�"
�
0���������
� z
#__inference_Rd_layer_call_fn_451411S7�4
-�*
 �
inputs���������

 
p
� "����������z
#__inference_Rd_layer_call_fn_451416S7�4
-�*
 �
inputs���������

 
p 
� "�����������
!__inference__wrapped_model_450402�
#$)*/04�1
*�'
%�"
input_pitch���������
� "���
&
Gain�
Gain���������
"
Rd�
Rd���������
"
p0�
p0���������

"
z0�
z0���������
�
C__inference_dense_1_layer_call_and_return_conditional_losses_451278\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
(__inference_dense_1_layer_call_fn_451285O/�,
%�"
 �
inputs���������
� "�����������
C__inference_dense_2_layer_call_and_return_conditional_losses_451296\#$/�,
%�"
 �
inputs���������
� "%�"
�
0��������� 
� {
(__inference_dense_2_layer_call_fn_451303O#$/�,
%�"
 �
inputs���������
� "���������� �
C__inference_dense_3_layer_call_and_return_conditional_losses_451314\)*/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������*
� {
(__inference_dense_3_layer_call_fn_451321O)*/�,
%�"
 �
inputs��������� 
� "����������*�
C__inference_dense_4_layer_call_and_return_conditional_losses_451331\/0/�,
%�"
 �
inputs���������*
� "%�"
�
0���������*
� {
(__inference_dense_4_layer_call_fn_451338O/0/�,
%�"
 �
inputs���������*
� "����������*�
A__inference_dense_layer_call_and_return_conditional_losses_451260\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� y
&__inference_dense_layer_call_fn_451267O/�,
%�"
 �
inputs���������
� "�����������
B__inference_lambda_layer_call_and_return_conditional_losses_451234`7�4
-�*
 �
inputs���������

 
p
� "%�"
�
0���������
� �
B__inference_lambda_layer_call_and_return_conditional_losses_451240`7�4
-�*
 �
inputs���������

 
p 
� "%�"
�
0���������
� ~
'__inference_lambda_layer_call_fn_451245S7�4
-�*
 �
inputs���������

 
p
� "����������~
'__inference_lambda_layer_call_fn_451250S7�4
-�*
 �
inputs���������

 
p 
� "�����������
A__inference_model_layer_call_and_return_conditional_losses_450788�
#$)*/0<�9
2�/
%�"
input_pitch���������
p

 
� "���
�|
�
0/0���������
�
0/1���������
�
0/2���������

�
0/3���������

� �
A__inference_model_layer_call_and_return_conditional_losses_450819�
#$)*/0<�9
2�/
%�"
input_pitch���������
p 

 
� "���
�|
�
0/0���������
�
0/1���������
�
0/2���������

�
0/3���������

� �
A__inference_model_layer_call_and_return_conditional_losses_451070�
#$)*/07�4
-�*
 �
inputs���������
p

 
� "���
�|
�
0/0���������
�
0/1���������
�
0/2���������

�
0/3���������

� �
A__inference_model_layer_call_and_return_conditional_losses_451186�
#$)*/07�4
-�*
 �
inputs���������
p 

 
� "���
�|
�
0/0���������
�
0/1���������
�
0/2���������

�
0/3���������

� �
&__inference_model_layer_call_fn_450872�
#$)*/0<�9
2�/
%�"
input_pitch���������
p

 
� "w�t
�
0���������
�
1���������
�
2���������

�
3���������
�
&__inference_model_layer_call_fn_450924�
#$)*/0<�9
2�/
%�"
input_pitch���������
p 

 
� "w�t
�
0���������
�
1���������
�
2���������

�
3���������
�
&__inference_model_layer_call_fn_451207�
#$)*/07�4
-�*
 �
inputs���������
p

 
� "w�t
�
0���������
�
1���������
�
2���������

�
3���������
�
&__inference_model_layer_call_fn_451228�
#$)*/07�4
-�*
 �
inputs���������
p 

 
� "w�t
�
0���������
�
1���������
�
2���������

�
3���������
�
>__inference_p0_layer_call_and_return_conditional_losses_451447`7�4
-�*
 �
inputs���������

 
p
� "%�"
�
0���������

� �
>__inference_p0_layer_call_and_return_conditional_losses_451478`7�4
-�*
 �
inputs���������

 
p 
� "%�"
�
0���������

� z
#__inference_p0_layer_call_fn_451483S7�4
-�*
 �
inputs���������

 
p
� "����������
z
#__inference_p0_layer_call_fn_451488S7�4
-�*
 �
inputs���������

 
p 
� "����������
�
$__inference_signature_wrapper_450954�
#$)*/0C�@
� 
9�6
4
input_pitch%�"
input_pitch���������"���
&
Gain�
Gain���������
"
Rd�
Rd���������
"
p0�
p0���������

"
z0�
z0���������
�
G__inference_slice_layer_layer_call_and_return_conditional_losses_451361�.�+
$�!
�
input���������*
� "���
�|
�
0/0���������
�
0/1���������
�
0/2���������
�
0/3���������
� �
,__inference_slice_layer_layer_call_fn_451372�.�+
$�!
�
input���������*
� "w�t
�
0���������
�
1���������
�
2���������
�
3����������
>__inference_z0_layer_call_and_return_conditional_losses_451519`7�4
-�*
 �
inputs���������

 
p
� "%�"
�
0���������

� �
>__inference_z0_layer_call_and_return_conditional_losses_451550`7�4
-�*
 �
inputs���������

 
p 
� "%�"
�
0���������

� z
#__inference_z0_layer_call_fn_451555S7�4
-�*
 �
inputs���������

 
p
� "����������
z
#__inference_z0_layer_call_fn_451560S7�4
-�*
 �
inputs���������

 
p 
� "����������
