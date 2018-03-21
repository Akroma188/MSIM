#include "__cf_popul.h"
#include <math.h>
#include "popul_acc.h"
#include "popul_acc_private.h"
#include <stdio.h>
#include "slexec_vm_simstruct_bridge.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_lookup_functions.h"
#include "simstruc.h"
#include "fixedpoint.h"
#define CodeFormat S-Function
#define AccDefine1 Accelerator_S-Function
static void mdlOutputs ( SimStruct * S , int_T tid ) { o1xh145t0o * _rtB ;
ieus1lybjf * _rtX ; duymdaka0x * _rtDW ; _rtDW = ( ( duymdaka0x * )
ssGetRootDWork ( S ) ) ; _rtX = ( ( ieus1lybjf * ) ssGetContStates ( S ) ) ;
_rtB = ( ( o1xh145t0o * ) _ssGetModelBlockIO ( S ) ) ; _rtB -> oyx5yfdzev =
_rtX -> ezl4vf4mon ; ssCallAccelRunBlock ( S , 0 , 1 , SS_CALL_MDL_OUTPUTS )
; _rtB -> bdd1xty4ee = _rtX -> mrddyjksun ; ssCallAccelRunBlock ( S , 0 , 3 ,
SS_CALL_MDL_OUTPUTS ) ; _rtB -> hduevjx32i = - _rtB -> oyx5yfdzev - _rtB ->
oyx5yfdzev * _rtB -> bdd1xty4ee ; _rtB -> c0vdjcdysr = _rtB -> oyx5yfdzev *
_rtB -> bdd1xty4ee + _rtB -> bdd1xty4ee ; _rtB -> blxzvy5wnp = ssGetT ( S ) ;
ssCallAccelRunBlock ( S , 0 , 7 , SS_CALL_MDL_OUTPUTS ) ; UNUSED_PARAMETER (
tid ) ; }
#define MDL_UPDATE
static void mdlUpdate ( SimStruct * S , int_T tid ) { UNUSED_PARAMETER ( tid
) ; }
#define MDL_DERIVATIVES
static void mdlDerivatives ( SimStruct * S ) { o1xh145t0o * _rtB ; hpwuqye3z2
* _rtXdot ; _rtXdot = ( ( hpwuqye3z2 * ) ssGetdX ( S ) ) ; _rtB = ( (
o1xh145t0o * ) _ssGetModelBlockIO ( S ) ) ; _rtXdot -> ezl4vf4mon = _rtB ->
hduevjx32i ; _rtXdot -> mrddyjksun = _rtB -> c0vdjcdysr ; } static void
mdlInitializeSizes ( SimStruct * S ) { ssSetChecksumVal ( S , 0 , 2298306949U
) ; ssSetChecksumVal ( S , 1 , 2992139631U ) ; ssSetChecksumVal ( S , 2 ,
1183135254U ) ; ssSetChecksumVal ( S , 3 , 898235592U ) ; { mxArray *
slVerStructMat = NULL ; mxArray * slStrMat = mxCreateString ( "simulink" ) ;
char slVerChar [ 10 ] ; int status = mexCallMATLAB ( 1 , & slVerStructMat , 1
, & slStrMat , "ver" ) ; if ( status == 0 ) { mxArray * slVerMat = mxGetField
( slVerStructMat , 0 , "Version" ) ; if ( slVerMat == NULL ) { status = 1 ; }
else { status = mxGetString ( slVerMat , slVerChar , 10 ) ; } }
mxDestroyArray ( slStrMat ) ; mxDestroyArray ( slVerStructMat ) ; if ( (
status == 1 ) || ( strcmp ( slVerChar , "8.7" ) != 0 ) ) { return ; } }
ssSetOptions ( S , SS_OPTION_EXCEPTION_FREE_CODE ) ; if ( ssGetSizeofDWork (
S ) != sizeof ( duymdaka0x ) ) { ssSetErrorStatus ( S ,
"Unexpected error: Internal DWork sizes do "
"not match for accelerator mex file." ) ; } if ( ssGetSizeofGlobalBlockIO ( S
) != sizeof ( o1xh145t0o ) ) { ssSetErrorStatus ( S ,
"Unexpected error: Internal BlockIO sizes do "
"not match for accelerator mex file." ) ; } { int ssSizeofParams ;
ssGetSizeofParams ( S , & ssSizeofParams ) ; if ( ssSizeofParams != sizeof (
a5axdcacmv ) ) { static char msg [ 256 ] ; sprintf ( msg ,
"Unexpected error: Internal Parameters sizes do "
"not match for accelerator mex file." ) ; } } _ssSetModelRtp ( S , ( real_T *
) & nmpjag1zkr ) ; } static void mdlInitializeSampleTimes ( SimStruct * S ) {
} static void mdlTerminate ( SimStruct * S ) { }
#include "simulink.c"
