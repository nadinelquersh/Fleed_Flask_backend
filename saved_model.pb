��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028��
�
sequential_3/dense_7/biasVarHandleOp*
_output_shapes
: **

debug_namesequential_3/dense_7/bias/*
dtype0*
shape:**
shared_namesequential_3/dense_7/bias
�
-sequential_3/dense_7/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_7/bias*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpsequential_3/dense_7/bias*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
sequential_3/dense_7/kernelVarHandleOp*
_output_shapes
: *,

debug_namesequential_3/dense_7/kernel/*
dtype0*
shape:	�*,
shared_namesequential_3/dense_7/kernel
�
/sequential_3/dense_7/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_7/kernel*
_output_shapes
:	�*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpsequential_3/dense_7/kernel*
_class
loc:@Variable_1*
_output_shapes
:	�*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:	�*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	�*
dtype0
�
sequential_3/dense_6/biasVarHandleOp*
_output_shapes
: **

debug_namesequential_3/dense_6/bias/*
dtype0*
shape:�**
shared_namesequential_3/dense_6/bias
�
-sequential_3/dense_6/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_6/bias*
_output_shapes	
:�*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOpsequential_3/dense_6/bias*
_class
loc:@Variable_2*
_output_shapes	
:�*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:�*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
f
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes	
:�*
dtype0
�
sequential_3/dense_6/kernelVarHandleOp*
_output_shapes
: *,

debug_namesequential_3/dense_6/kernel/*
dtype0*
shape:
��*,
shared_namesequential_3/dense_6/kernel
�
/sequential_3/dense_6/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_6/kernel* 
_output_shapes
:
��*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpsequential_3/dense_6/kernel*
_class
loc:@Variable_3* 
_output_shapes
:
��*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:
��*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
k
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3* 
_output_shapes
:
��*
dtype0
�
sequential_3/conv2d_11/biasVarHandleOp*
_output_shapes
: *,

debug_namesequential_3/conv2d_11/bias/*
dtype0*
shape:�*,
shared_namesequential_3/conv2d_11/bias
�
/sequential_3/conv2d_11/bias/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_11/bias*
_output_shapes	
:�*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpsequential_3/conv2d_11/bias*
_class
loc:@Variable_4*
_output_shapes	
:�*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:�*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
f
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes	
:�*
dtype0
�
sequential_3/conv2d_11/kernelVarHandleOp*
_output_shapes
: *.

debug_name sequential_3/conv2d_11/kernel/*
dtype0*
shape:@�*.
shared_namesequential_3/conv2d_11/kernel
�
1sequential_3/conv2d_11/kernel/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_11/kernel*'
_output_shapes
:@�*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOpsequential_3/conv2d_11/kernel*
_class
loc:@Variable_5*'
_output_shapes
:@�*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:@�*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
r
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*'
_output_shapes
:@�*
dtype0
�
sequential_3/conv2d_10/biasVarHandleOp*
_output_shapes
: *,

debug_namesequential_3/conv2d_10/bias/*
dtype0*
shape:@*,
shared_namesequential_3/conv2d_10/bias
�
/sequential_3/conv2d_10/bias/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_10/bias*
_output_shapes
:@*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpsequential_3/conv2d_10/bias*
_class
loc:@Variable_6*
_output_shapes
:@*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:@*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:@*
dtype0
�
sequential_3/conv2d_10/kernelVarHandleOp*
_output_shapes
: *.

debug_name sequential_3/conv2d_10/kernel/*
dtype0*
shape: @*.
shared_namesequential_3/conv2d_10/kernel
�
1sequential_3/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_10/kernel*&
_output_shapes
: @*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpsequential_3/conv2d_10/kernel*
_class
loc:@Variable_7*&
_output_shapes
: @*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape: @*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
q
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*&
_output_shapes
: @*
dtype0
�
sequential_3/conv2d_9/biasVarHandleOp*
_output_shapes
: *+

debug_namesequential_3/conv2d_9/bias/*
dtype0*
shape: *+
shared_namesequential_3/conv2d_9/bias
�
.sequential_3/conv2d_9/bias/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_9/bias*
_output_shapes
: *
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpsequential_3/conv2d_9/bias*
_class
loc:@Variable_8*
_output_shapes
: *
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape: *
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
e
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
: *
dtype0
�
sequential_3/conv2d_9/kernelVarHandleOp*
_output_shapes
: *-

debug_namesequential_3/conv2d_9/kernel/*
dtype0*
shape: *-
shared_namesequential_3/conv2d_9/kernel
�
0sequential_3/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_9/kernel*&
_output_shapes
: *
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpsequential_3/conv2d_9/kernel*
_class
loc:@Variable_9*&
_output_shapes
: *
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape: *
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
q
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*&
_output_shapes
: *
dtype0
�
adam/learning_rateVarHandleOp*
_output_shapes
: *#

debug_nameadam/learning_rate/*
dtype0*
shape: *#
shared_nameadam/learning_rate
q
&adam/learning_rate/Read/ReadVariableOpReadVariableOpadam/learning_rate*
_output_shapes
: *
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpadam/learning_rate*
_class
loc:@Variable_10*
_output_shapes
: *
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape: *
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
c
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
: *
dtype0
�
adam/iterationVarHandleOp*
_output_shapes
: *

debug_nameadam/iteration/*
dtype0	*
shape: *
shared_nameadam/iteration
i
"adam/iteration/Read/ReadVariableOpReadVariableOpadam/iteration*
_output_shapes
: *
dtype0	
�
&Variable_11/Initializer/ReadVariableOpReadVariableOpadam/iteration*
_class
loc:@Variable_11*
_output_shapes
: *
dtype0	
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0	*
shape: *
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0	
c
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
: *
dtype0	
�
serving_default_inputsPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputssequential_3/conv2d_9/kernelsequential_3/conv2d_9/biassequential_3/conv2d_10/kernelsequential_3/conv2d_10/biassequential_3/conv2d_11/kernelsequential_3/conv2d_11/biassequential_3/dense_6/kernelsequential_3/dense_6/biassequential_3/dense_7/kernelsequential_3/dense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU 2J 8� �J *:
f5R3
1__inference_signature_wrapper_serving_default_221

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
_functional
	optimizer
_default_save_signature
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids
_losses_override
	_layers

_build_shapes_dict

signatures*
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
output_names
_default_save_signature*
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
_velocities*

trace_0* 
* 
* 
* 
* 
* 
R
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10*
* 

)serving_default* 
* 
* 
* 
* 
* 
R
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10*
R
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10*
* 
* 

*trace_0* 

0
1*
* 
* 
UO
VARIABLE_VALUEVariable_110optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEVariable_103optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
]
+_inbound_nodes
,_outbound_nodes
-_losses
.	_loss_ids
/_losses_override* 
�
0_kernel
1bias
2_inbound_nodes
3_outbound_nodes
4_losses
5	_loss_ids
6_losses_override
7_build_shapes_dict*
]
8_inbound_nodes
9_outbound_nodes
:_losses
;	_loss_ids
<_losses_override* 
�
=_kernel
>bias
?_inbound_nodes
@_outbound_nodes
A_losses
B	_loss_ids
C_losses_override
D_build_shapes_dict*
]
E_inbound_nodes
F_outbound_nodes
G_losses
H	_loss_ids
I_losses_override* 
�
J_kernel
Kbias
L_inbound_nodes
M_outbound_nodes
N_losses
O	_loss_ids
P_losses_override
Q_build_shapes_dict*
]
R_inbound_nodes
S_outbound_nodes
T_losses
U	_loss_ids
V_losses_override* 
u
W_inbound_nodes
X_outbound_nodes
Y_losses
Z	_loss_ids
[_losses_override
\_build_shapes_dict* 
�
]_kernel
^bias
__inbound_nodes
`_outbound_nodes
a_losses
b	_loss_ids
c_losses_override
d_build_shapes_dict*
]
e_inbound_nodes
f_outbound_nodes
g_losses
h	_loss_ids
i_losses_override* 
�
j_kernel
kbias
l_inbound_nodes
m_outbound_nodes
n_losses
o	_loss_ids
p_losses_override
q_build_shapes_dict*
* 
* 
* 
* 
* 
* 
* 
PJ
VARIABLE_VALUE
Variable_9,_layers/1/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE
Variable_8)_layers/1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
PJ
VARIABLE_VALUE
Variable_7,_layers/3/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE
Variable_6)_layers/3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
PJ
VARIABLE_VALUE
Variable_5,_layers/5/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE
Variable_4)_layers/5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
PJ
VARIABLE_VALUE
Variable_3,_layers/8/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE
Variable_2)_layers/8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
QK
VARIABLE_VALUE
Variable_1-_layers/10/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable*_layers/10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *%
f R
__inference__traced_save_416
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *(
f#R!
__inference__traced_restore_461��
�
�
1__inference_signature_wrapper_serving_default_221

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU 2J 8� �J *(
f#R!
__inference_serving_default_195o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:#


_user_specified_name217:#	

_user_specified_name215:#

_user_specified_name213:#

_user_specified_name211:#

_user_specified_name209:#

_user_specified_name207:#

_user_specified_name205:#

_user_specified_name203:#

_user_specified_name201:#

_user_specified_name199:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�S
�

__inference_serving_default_274

inputsW
=functional_9_1_conv2d_9_1_convolution_readvariableop_resource: G
9functional_9_1_conv2d_9_1_reshape_readvariableop_resource: X
>functional_9_1_conv2d_10_1_convolution_readvariableop_resource: @H
:functional_9_1_conv2d_10_1_reshape_readvariableop_resource:@Y
>functional_9_1_conv2d_11_1_convolution_readvariableop_resource:@�I
:functional_9_1_conv2d_11_1_reshape_readvariableop_resource:	�I
5functional_9_1_dense_6_1_cast_readvariableop_resource:
��G
8functional_9_1_dense_6_1_biasadd_readvariableop_resource:	�H
5functional_9_1_dense_7_1_cast_readvariableop_resource:	�F
8functional_9_1_dense_7_1_biasadd_readvariableop_resource:
identity��1functional_9_1/conv2d_10_1/Reshape/ReadVariableOp�5functional_9_1/conv2d_10_1/convolution/ReadVariableOp�1functional_9_1/conv2d_11_1/Reshape/ReadVariableOp�5functional_9_1/conv2d_11_1/convolution/ReadVariableOp�0functional_9_1/conv2d_9_1/Reshape/ReadVariableOp�4functional_9_1/conv2d_9_1/convolution/ReadVariableOp�/functional_9_1/dense_6_1/BiasAdd/ReadVariableOp�,functional_9_1/dense_6_1/Cast/ReadVariableOp�/functional_9_1/dense_7_1/BiasAdd/ReadVariableOp�,functional_9_1/dense_7_1/Cast/ReadVariableOp�
4functional_9_1/conv2d_9_1/convolution/ReadVariableOpReadVariableOp=functional_9_1_conv2d_9_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
%functional_9_1/conv2d_9_1/convolutionConv2Dinputs<functional_9_1/conv2d_9_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
�
0functional_9_1/conv2d_9_1/Reshape/ReadVariableOpReadVariableOp9functional_9_1_conv2d_9_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0�
'functional_9_1/conv2d_9_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
!functional_9_1/conv2d_9_1/ReshapeReshape8functional_9_1/conv2d_9_1/Reshape/ReadVariableOp:value:00functional_9_1/conv2d_9_1/Reshape/shape:output:0*
T0*&
_output_shapes
: }
!functional_9_1/conv2d_9_1/SqueezeSqueeze*functional_9_1/conv2d_9_1/Reshape:output:0*
T0*
_output_shapes
: �
!functional_9_1/conv2d_9_1/BiasAddBiasAdd.functional_9_1/conv2d_9_1/convolution:output:0*functional_9_1/conv2d_9_1/Squeeze:output:0*
T0*/
_output_shapes
:���������@@ �
functional_9_1/conv2d_9_1/ReluRelu*functional_9_1/conv2d_9_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ �
*functional_9_1/max_pooling2d_9_1/MaxPool2dMaxPool,functional_9_1/conv2d_9_1/Relu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
�
5functional_9_1/conv2d_10_1/convolution/ReadVariableOpReadVariableOp>functional_9_1_conv2d_10_1_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&functional_9_1/conv2d_10_1/convolutionConv2D3functional_9_1/max_pooling2d_9_1/MaxPool2d:output:0=functional_9_1/conv2d_10_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
1functional_9_1/conv2d_10_1/Reshape/ReadVariableOpReadVariableOp:functional_9_1_conv2d_10_1_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0�
(functional_9_1/conv2d_10_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
"functional_9_1/conv2d_10_1/ReshapeReshape9functional_9_1/conv2d_10_1/Reshape/ReadVariableOp:value:01functional_9_1/conv2d_10_1/Reshape/shape:output:0*
T0*&
_output_shapes
:@
"functional_9_1/conv2d_10_1/SqueezeSqueeze+functional_9_1/conv2d_10_1/Reshape:output:0*
T0*
_output_shapes
:@�
"functional_9_1/conv2d_10_1/BiasAddBiasAdd/functional_9_1/conv2d_10_1/convolution:output:0+functional_9_1/conv2d_10_1/Squeeze:output:0*
T0*/
_output_shapes
:���������@�
functional_9_1/conv2d_10_1/ReluRelu+functional_9_1/conv2d_10_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
+functional_9_1/max_pooling2d_10_1/MaxPool2dMaxPool-functional_9_1/conv2d_10_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
�
5functional_9_1/conv2d_11_1/convolution/ReadVariableOpReadVariableOp>functional_9_1_conv2d_11_1_convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&functional_9_1/conv2d_11_1/convolutionConv2D4functional_9_1/max_pooling2d_10_1/MaxPool2d:output:0=functional_9_1/conv2d_11_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
1functional_9_1/conv2d_11_1/Reshape/ReadVariableOpReadVariableOp:functional_9_1_conv2d_11_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(functional_9_1/conv2d_11_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
"functional_9_1/conv2d_11_1/ReshapeReshape9functional_9_1/conv2d_11_1/Reshape/ReadVariableOp:value:01functional_9_1/conv2d_11_1/Reshape/shape:output:0*
T0*'
_output_shapes
:��
"functional_9_1/conv2d_11_1/SqueezeSqueeze+functional_9_1/conv2d_11_1/Reshape:output:0*
T0*
_output_shapes	
:��
"functional_9_1/conv2d_11_1/BiasAddBiasAdd/functional_9_1/conv2d_11_1/convolution:output:0+functional_9_1/conv2d_11_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
functional_9_1/conv2d_11_1/ReluRelu+functional_9_1/conv2d_11_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
+functional_9_1/max_pooling2d_11_1/MaxPool2dMaxPool-functional_9_1/conv2d_11_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
y
(functional_9_1/flatten_3_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"functional_9_1/flatten_3_1/ReshapeReshape4functional_9_1/max_pooling2d_11_1/MaxPool2d:output:01functional_9_1/flatten_3_1/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
,functional_9_1/dense_6_1/Cast/ReadVariableOpReadVariableOp5functional_9_1_dense_6_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
functional_9_1/dense_6_1/MatMulMatMul+functional_9_1/flatten_3_1/Reshape:output:04functional_9_1/dense_6_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/functional_9_1/dense_6_1/BiasAdd/ReadVariableOpReadVariableOp8functional_9_1_dense_6_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 functional_9_1/dense_6_1/BiasAddBiasAdd)functional_9_1/dense_6_1/MatMul:product:07functional_9_1/dense_6_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
functional_9_1/dense_6_1/ReluRelu)functional_9_1/dense_6_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,functional_9_1/dense_7_1/Cast/ReadVariableOpReadVariableOp5functional_9_1_dense_7_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
functional_9_1/dense_7_1/MatMulMatMul+functional_9_1/dense_6_1/Relu:activations:04functional_9_1/dense_7_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/functional_9_1/dense_7_1/BiasAdd/ReadVariableOpReadVariableOp8functional_9_1_dense_7_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 functional_9_1/dense_7_1/BiasAddBiasAdd)functional_9_1/dense_7_1/MatMul:product:07functional_9_1/dense_7_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 functional_9_1/dense_7_1/SoftmaxSoftmax)functional_9_1/dense_7_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������y
IdentityIdentity*functional_9_1/dense_7_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp2^functional_9_1/conv2d_10_1/Reshape/ReadVariableOp6^functional_9_1/conv2d_10_1/convolution/ReadVariableOp2^functional_9_1/conv2d_11_1/Reshape/ReadVariableOp6^functional_9_1/conv2d_11_1/convolution/ReadVariableOp1^functional_9_1/conv2d_9_1/Reshape/ReadVariableOp5^functional_9_1/conv2d_9_1/convolution/ReadVariableOp0^functional_9_1/dense_6_1/BiasAdd/ReadVariableOp-^functional_9_1/dense_6_1/Cast/ReadVariableOp0^functional_9_1/dense_7_1/BiasAdd/ReadVariableOp-^functional_9_1/dense_7_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2f
1functional_9_1/conv2d_10_1/Reshape/ReadVariableOp1functional_9_1/conv2d_10_1/Reshape/ReadVariableOp2n
5functional_9_1/conv2d_10_1/convolution/ReadVariableOp5functional_9_1/conv2d_10_1/convolution/ReadVariableOp2f
1functional_9_1/conv2d_11_1/Reshape/ReadVariableOp1functional_9_1/conv2d_11_1/Reshape/ReadVariableOp2n
5functional_9_1/conv2d_11_1/convolution/ReadVariableOp5functional_9_1/conv2d_11_1/convolution/ReadVariableOp2d
0functional_9_1/conv2d_9_1/Reshape/ReadVariableOp0functional_9_1/conv2d_9_1/Reshape/ReadVariableOp2l
4functional_9_1/conv2d_9_1/convolution/ReadVariableOp4functional_9_1/conv2d_9_1/convolution/ReadVariableOp2b
/functional_9_1/dense_6_1/BiasAdd/ReadVariableOp/functional_9_1/dense_6_1/BiasAdd/ReadVariableOp2\
,functional_9_1/dense_6_1/Cast/ReadVariableOp,functional_9_1/dense_6_1/Cast/ReadVariableOp2b
/functional_9_1/dense_7_1/BiasAdd/ReadVariableOp/functional_9_1/dense_7_1/BiasAdd/ReadVariableOp2\
,functional_9_1/dense_7_1/Cast/ReadVariableOp,functional_9_1/dense_7_1/Cast/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�_
�

__inference__traced_save_416
file_prefix,
"read_disablecopyonread_variable_11:	 .
$read_1_disablecopyonread_variable_10: =
#read_2_disablecopyonread_variable_9: 1
#read_3_disablecopyonread_variable_8: =
#read_4_disablecopyonread_variable_7: @1
#read_5_disablecopyonread_variable_6:@>
#read_6_disablecopyonread_variable_5:@�2
#read_7_disablecopyonread_variable_4:	�7
#read_8_disablecopyonread_variable_3:
��2
#read_9_disablecopyonread_variable_2:	�7
$read_10_disablecopyonread_variable_1:	�0
"read_11_disablecopyonread_variable:
savev2_const
identity_25��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_11*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_11^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0	R
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_10*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_10^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_2/DisableCopyOnReadDisableCopyOnRead#read_2_disablecopyonread_variable_9*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp#read_2_disablecopyonread_variable_9^Read_2/DisableCopyOnRead*&
_output_shapes
: *
dtype0f

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: h
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_variable_8*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_variable_8^Read_3/DisableCopyOnRead*
_output_shapes
: *
dtype0Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_variable_7*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_variable_7^Read_4/DisableCopyOnRead*&
_output_shapes
: @*
dtype0f

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*&
_output_shapes
: @k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: @h
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_variable_6*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_variable_6^Read_5/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@h
Read_6/DisableCopyOnReadDisableCopyOnRead#read_6_disablecopyonread_variable_5*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp#read_6_disablecopyonread_variable_5^Read_6/DisableCopyOnRead*'
_output_shapes
:@�*
dtype0h
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�n
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�h
Read_7/DisableCopyOnReadDisableCopyOnRead#read_7_disablecopyonread_variable_4*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp#read_7_disablecopyonread_variable_4^Read_7/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�h
Read_8/DisableCopyOnReadDisableCopyOnRead#read_8_disablecopyonread_variable_3*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp#read_8_disablecopyonread_variable_3^Read_8/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��h
Read_9/DisableCopyOnReadDisableCopyOnRead#read_9_disablecopyonread_variable_2*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp#read_9_disablecopyonread_variable_2^Read_9/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_10/DisableCopyOnReadDisableCopyOnRead$read_10_disablecopyonread_variable_1*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp$read_10_disablecopyonread_variable_1^Read_10/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Read_11/DisableCopyOnReadDisableCopyOnRead"read_11_disablecopyonread_variable*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp"read_11_disablecopyonread_variable^Read_11/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB,_layers/1/_kernel/.ATTRIBUTES/VARIABLE_VALUEB)_layers/1/bias/.ATTRIBUTES/VARIABLE_VALUEB,_layers/3/_kernel/.ATTRIBUTES/VARIABLE_VALUEB)_layers/3/bias/.ATTRIBUTES/VARIABLE_VALUEB,_layers/5/_kernel/.ATTRIBUTES/VARIABLE_VALUEB)_layers/5/bias/.ATTRIBUTES/VARIABLE_VALUEB,_layers/8/_kernel/.ATTRIBUTES/VARIABLE_VALUEB)_layers/8/bias/.ATTRIBUTES/VARIABLE_VALUEB-_layers/10/_kernel/.ATTRIBUTES/VARIABLE_VALUEB*_layers/10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_24Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_25IdentityIdentity_24:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_25Identity_25:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*
&
$
_user_specified_name
Variable_2:*	&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�S
�

__inference_serving_default_195

inputsW
=sequential_3_1_conv2d_9_1_convolution_readvariableop_resource: G
9sequential_3_1_conv2d_9_1_reshape_readvariableop_resource: X
>sequential_3_1_conv2d_10_1_convolution_readvariableop_resource: @H
:sequential_3_1_conv2d_10_1_reshape_readvariableop_resource:@Y
>sequential_3_1_conv2d_11_1_convolution_readvariableop_resource:@�I
:sequential_3_1_conv2d_11_1_reshape_readvariableop_resource:	�I
5sequential_3_1_dense_6_1_cast_readvariableop_resource:
��G
8sequential_3_1_dense_6_1_biasadd_readvariableop_resource:	�H
5sequential_3_1_dense_7_1_cast_readvariableop_resource:	�F
8sequential_3_1_dense_7_1_biasadd_readvariableop_resource:
identity��1sequential_3_1/conv2d_10_1/Reshape/ReadVariableOp�5sequential_3_1/conv2d_10_1/convolution/ReadVariableOp�1sequential_3_1/conv2d_11_1/Reshape/ReadVariableOp�5sequential_3_1/conv2d_11_1/convolution/ReadVariableOp�0sequential_3_1/conv2d_9_1/Reshape/ReadVariableOp�4sequential_3_1/conv2d_9_1/convolution/ReadVariableOp�/sequential_3_1/dense_6_1/BiasAdd/ReadVariableOp�,sequential_3_1/dense_6_1/Cast/ReadVariableOp�/sequential_3_1/dense_7_1/BiasAdd/ReadVariableOp�,sequential_3_1/dense_7_1/Cast/ReadVariableOp�
4sequential_3_1/conv2d_9_1/convolution/ReadVariableOpReadVariableOp=sequential_3_1_conv2d_9_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
%sequential_3_1/conv2d_9_1/convolutionConv2Dinputs<sequential_3_1/conv2d_9_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
�
0sequential_3_1/conv2d_9_1/Reshape/ReadVariableOpReadVariableOp9sequential_3_1_conv2d_9_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0�
'sequential_3_1/conv2d_9_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
!sequential_3_1/conv2d_9_1/ReshapeReshape8sequential_3_1/conv2d_9_1/Reshape/ReadVariableOp:value:00sequential_3_1/conv2d_9_1/Reshape/shape:output:0*
T0*&
_output_shapes
: }
!sequential_3_1/conv2d_9_1/SqueezeSqueeze*sequential_3_1/conv2d_9_1/Reshape:output:0*
T0*
_output_shapes
: �
!sequential_3_1/conv2d_9_1/BiasAddBiasAdd.sequential_3_1/conv2d_9_1/convolution:output:0*sequential_3_1/conv2d_9_1/Squeeze:output:0*
T0*/
_output_shapes
:���������@@ �
sequential_3_1/conv2d_9_1/ReluRelu*sequential_3_1/conv2d_9_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ �
*sequential_3_1/max_pooling2d_9_1/MaxPool2dMaxPool,sequential_3_1/conv2d_9_1/Relu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
�
5sequential_3_1/conv2d_10_1/convolution/ReadVariableOpReadVariableOp>sequential_3_1_conv2d_10_1_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&sequential_3_1/conv2d_10_1/convolutionConv2D3sequential_3_1/max_pooling2d_9_1/MaxPool2d:output:0=sequential_3_1/conv2d_10_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
1sequential_3_1/conv2d_10_1/Reshape/ReadVariableOpReadVariableOp:sequential_3_1_conv2d_10_1_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0�
(sequential_3_1/conv2d_10_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
"sequential_3_1/conv2d_10_1/ReshapeReshape9sequential_3_1/conv2d_10_1/Reshape/ReadVariableOp:value:01sequential_3_1/conv2d_10_1/Reshape/shape:output:0*
T0*&
_output_shapes
:@
"sequential_3_1/conv2d_10_1/SqueezeSqueeze+sequential_3_1/conv2d_10_1/Reshape:output:0*
T0*
_output_shapes
:@�
"sequential_3_1/conv2d_10_1/BiasAddBiasAdd/sequential_3_1/conv2d_10_1/convolution:output:0+sequential_3_1/conv2d_10_1/Squeeze:output:0*
T0*/
_output_shapes
:���������@�
sequential_3_1/conv2d_10_1/ReluRelu+sequential_3_1/conv2d_10_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
+sequential_3_1/max_pooling2d_10_1/MaxPool2dMaxPool-sequential_3_1/conv2d_10_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
�
5sequential_3_1/conv2d_11_1/convolution/ReadVariableOpReadVariableOp>sequential_3_1_conv2d_11_1_convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&sequential_3_1/conv2d_11_1/convolutionConv2D4sequential_3_1/max_pooling2d_10_1/MaxPool2d:output:0=sequential_3_1/conv2d_11_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
1sequential_3_1/conv2d_11_1/Reshape/ReadVariableOpReadVariableOp:sequential_3_1_conv2d_11_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(sequential_3_1/conv2d_11_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
"sequential_3_1/conv2d_11_1/ReshapeReshape9sequential_3_1/conv2d_11_1/Reshape/ReadVariableOp:value:01sequential_3_1/conv2d_11_1/Reshape/shape:output:0*
T0*'
_output_shapes
:��
"sequential_3_1/conv2d_11_1/SqueezeSqueeze+sequential_3_1/conv2d_11_1/Reshape:output:0*
T0*
_output_shapes	
:��
"sequential_3_1/conv2d_11_1/BiasAddBiasAdd/sequential_3_1/conv2d_11_1/convolution:output:0+sequential_3_1/conv2d_11_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
sequential_3_1/conv2d_11_1/ReluRelu+sequential_3_1/conv2d_11_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
+sequential_3_1/max_pooling2d_11_1/MaxPool2dMaxPool-sequential_3_1/conv2d_11_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
y
(sequential_3_1/flatten_3_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"sequential_3_1/flatten_3_1/ReshapeReshape4sequential_3_1/max_pooling2d_11_1/MaxPool2d:output:01sequential_3_1/flatten_3_1/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
,sequential_3_1/dense_6_1/Cast/ReadVariableOpReadVariableOp5sequential_3_1_dense_6_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_3_1/dense_6_1/MatMulMatMul+sequential_3_1/flatten_3_1/Reshape:output:04sequential_3_1/dense_6_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/sequential_3_1/dense_6_1/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_1_dense_6_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 sequential_3_1/dense_6_1/BiasAddBiasAdd)sequential_3_1/dense_6_1/MatMul:product:07sequential_3_1/dense_6_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_3_1/dense_6_1/ReluRelu)sequential_3_1/dense_6_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,sequential_3_1/dense_7_1/Cast/ReadVariableOpReadVariableOp5sequential_3_1_dense_7_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_3_1/dense_7_1/MatMulMatMul+sequential_3_1/dense_6_1/Relu:activations:04sequential_3_1/dense_7_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/sequential_3_1/dense_7_1/BiasAdd/ReadVariableOpReadVariableOp8sequential_3_1_dense_7_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 sequential_3_1/dense_7_1/BiasAddBiasAdd)sequential_3_1/dense_7_1/MatMul:product:07sequential_3_1/dense_7_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 sequential_3_1/dense_7_1/SoftmaxSoftmax)sequential_3_1/dense_7_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������y
IdentityIdentity*sequential_3_1/dense_7_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp2^sequential_3_1/conv2d_10_1/Reshape/ReadVariableOp6^sequential_3_1/conv2d_10_1/convolution/ReadVariableOp2^sequential_3_1/conv2d_11_1/Reshape/ReadVariableOp6^sequential_3_1/conv2d_11_1/convolution/ReadVariableOp1^sequential_3_1/conv2d_9_1/Reshape/ReadVariableOp5^sequential_3_1/conv2d_9_1/convolution/ReadVariableOp0^sequential_3_1/dense_6_1/BiasAdd/ReadVariableOp-^sequential_3_1/dense_6_1/Cast/ReadVariableOp0^sequential_3_1/dense_7_1/BiasAdd/ReadVariableOp-^sequential_3_1/dense_7_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2f
1sequential_3_1/conv2d_10_1/Reshape/ReadVariableOp1sequential_3_1/conv2d_10_1/Reshape/ReadVariableOp2n
5sequential_3_1/conv2d_10_1/convolution/ReadVariableOp5sequential_3_1/conv2d_10_1/convolution/ReadVariableOp2f
1sequential_3_1/conv2d_11_1/Reshape/ReadVariableOp1sequential_3_1/conv2d_11_1/Reshape/ReadVariableOp2n
5sequential_3_1/conv2d_11_1/convolution/ReadVariableOp5sequential_3_1/conv2d_11_1/convolution/ReadVariableOp2d
0sequential_3_1/conv2d_9_1/Reshape/ReadVariableOp0sequential_3_1/conv2d_9_1/Reshape/ReadVariableOp2l
4sequential_3_1/conv2d_9_1/convolution/ReadVariableOp4sequential_3_1/conv2d_9_1/convolution/ReadVariableOp2b
/sequential_3_1/dense_6_1/BiasAdd/ReadVariableOp/sequential_3_1/dense_6_1/BiasAdd/ReadVariableOp2\
,sequential_3_1/dense_6_1/Cast/ReadVariableOp,sequential_3_1/dense_6_1/Cast/ReadVariableOp2b
/sequential_3_1/dense_7_1/BiasAdd/ReadVariableOp/sequential_3_1/dense_7_1/BiasAdd/ReadVariableOp2\
,sequential_3_1/dense_7_1/Cast/ReadVariableOp,sequential_3_1/dense_7_1/Cast/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�9
�
__inference__traced_restore_461
file_prefix&
assignvariableop_variable_11:	 (
assignvariableop_1_variable_10: 7
assignvariableop_2_variable_9: +
assignvariableop_3_variable_8: 7
assignvariableop_4_variable_7: @+
assignvariableop_5_variable_6:@8
assignvariableop_6_variable_5:@�,
assignvariableop_7_variable_4:	�1
assignvariableop_8_variable_3:
��,
assignvariableop_9_variable_2:	�1
assignvariableop_10_variable_1:	�*
assignvariableop_11_variable:
identity_13��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB,_layers/1/_kernel/.ATTRIBUTES/VARIABLE_VALUEB)_layers/1/bias/.ATTRIBUTES/VARIABLE_VALUEB,_layers/3/_kernel/.ATTRIBUTES/VARIABLE_VALUEB)_layers/3/bias/.ATTRIBUTES/VARIABLE_VALUEB,_layers/5/_kernel/.ATTRIBUTES/VARIABLE_VALUEB)_layers/5/bias/.ATTRIBUTES/VARIABLE_VALUEB,_layers/8/_kernel/.ATTRIBUTES/VARIABLE_VALUEB)_layers/8/bias/.ATTRIBUTES/VARIABLE_VALUEB-_layers/10/_kernel/.ATTRIBUTES/VARIABLE_VALUEB*_layers/10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_11Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_10Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_9Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_8Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_7Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_6Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_5Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_4Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_3Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_2Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variableIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_13Identity_13:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*
&
$
_user_specified_name
Variable_2:*	&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
inputs9
serving_default_inputs:0�����������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict:�:
�
_functional
	optimizer
_default_save_signature
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids
_losses_override
	_layers

_build_shapes_dict

signatures"
_generic_user_object
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
output_names
_default_save_signature"
_generic_user_object
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
_velocities"
_generic_user_object
�
trace_02�
__inference_serving_default_195�
���
FullArgSpec
args�

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
annotations� *'�$
"������������ztrace_0
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
n
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10"
trackable_list_wrapper
 "
trackable_dict_wrapper
,
)serving_default"
signature_map
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
n
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10"
trackable_list_wrapper
n
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
*trace_02�
__inference_serving_default_274�
���
FullArgSpec
args�

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
annotations� *'�$
"������������z*trace_0
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:	 (2adam/iteration
: (2adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�B�
__inference_serving_default_195inputs"�
���
FullArgSpec
args�

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
annotations� *
 
y
+_inbound_nodes
,_outbound_nodes
-_losses
.	_loss_ids
/_losses_override"
_generic_user_object
�
0_kernel
1bias
2_inbound_nodes
3_outbound_nodes
4_losses
5	_loss_ids
6_losses_override
7_build_shapes_dict"
_generic_user_object
y
8_inbound_nodes
9_outbound_nodes
:_losses
;	_loss_ids
<_losses_override"
_generic_user_object
�
=_kernel
>bias
?_inbound_nodes
@_outbound_nodes
A_losses
B	_loss_ids
C_losses_override
D_build_shapes_dict"
_generic_user_object
y
E_inbound_nodes
F_outbound_nodes
G_losses
H	_loss_ids
I_losses_override"
_generic_user_object
�
J_kernel
Kbias
L_inbound_nodes
M_outbound_nodes
N_losses
O	_loss_ids
P_losses_override
Q_build_shapes_dict"
_generic_user_object
y
R_inbound_nodes
S_outbound_nodes
T_losses
U	_loss_ids
V_losses_override"
_generic_user_object
�
W_inbound_nodes
X_outbound_nodes
Y_losses
Z	_loss_ids
[_losses_override
\_build_shapes_dict"
_generic_user_object
�
]_kernel
^bias
__inbound_nodes
`_outbound_nodes
a_losses
b	_loss_ids
c_losses_override
d_build_shapes_dict"
_generic_user_object
y
e_inbound_nodes
f_outbound_nodes
g_losses
h	_loss_ids
i_losses_override"
_generic_user_object
�
j_kernel
kbias
l_inbound_nodes
m_outbound_nodes
n_losses
o	_loss_ids
p_losses_override
q_build_shapes_dict"
_generic_user_object
�B�
1__inference_signature_wrapper_serving_default_221inputs"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jinputs
kwonlydefaults
 
annotations� *
 
�B�
__inference_serving_default_274inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
8:6 (2sequential_3/conv2d_9/kernel
*:( (2sequential_3/conv2d_9/bias
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
trackable_dict_wrapper
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
9:7 @(2sequential_3/conv2d_10/kernel
+:)@(2sequential_3/conv2d_10/bias
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
trackable_dict_wrapper
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
::8@�(2sequential_3/conv2d_11/kernel
,:*�(2sequential_3/conv2d_11/bias
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
trackable_dict_wrapper
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
trackable_dict_wrapper
1:/
��(2sequential_3/dense_6/kernel
*:(�(2sequential_3/dense_6/bias
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
trackable_dict_wrapper
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
0:.	�(2sequential_3/dense_7/kernel
):'(2sequential_3/dense_7/bias
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
trackable_dict_wrapper�
__inference_serving_default_195j
01=>JK]^jk9�6
/�,
*�'
inputs�����������
� "!�
unknown����������
__inference_serving_default_274j
01=>JK]^jk9�6
/�,
*�'
inputs�����������
� "!�
unknown����������
1__inference_signature_wrapper_serving_default_221�
01=>JK]^jkC�@
� 
9�6
4
inputs*�'
inputs�����������"3�0
.
output_0"�
output_0���������