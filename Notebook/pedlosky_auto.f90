!----------------------------------------------------------------------
!----------------------------------------------------------------------
!  PEDLOSKY TWO-LAYER MODEL
!----------------------------------------------------------------------
!----------------------------------------------------------------------

FUNCTION H_FUNCTION(k, aa)

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: k
    DOUBLE PRECISION, INTENT(IN) :: aa
    DOUBLE PRECISION H_FUNCTION, Pi

    Pi=dacos(-1.D0)
    H_FUNCTION = ((k - 0.5d0) ** 2) / ((k - 0.5d0) ** 2 + (aa ** 2) / (4 * Pi ** 2))

END FUNCTION H_FUNCTION

FUNCTION G_FUNCTION(k, aa)

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: k
    DOUBLE PRECISION, INTENT(IN) :: aa
    DOUBLE PRECISION G_FUNCTION, H_FUNCTION, Pi

    Pi=dacos(-1.D0)
    G_FUNCTION = H_FUNCTION(k, aa) + ((aa ** 2) / (2 * Pi ** 2)) / ((k - 0.5d0) ** 2 + (aa ** 2) / (4 * Pi ** 2))

END FUNCTION G_FUNCTION

FUNCTION F_FUNCTION(k, aa, m)

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: k, m
    DOUBLE PRECISION, INTENT(IN) :: aa
    DOUBLE PRECISION F_FUNCTION, H_FUNCTION, Pi

    Pi=dacos(-1.D0)
    F_FUNCTION = ((2 * m ** 2) / Pi ** 2) * (H_FUNCTION(k, aa) / ((k - 0.5d0) ** 2 - m ** 2) ** 2)

END FUNCTION F_FUNCTION

FUNCTION SUM_PART(X, kc, aa, m)

    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: X(*), aa
    INTEGER, INTENT(IN) :: kc, m
    INTEGER i
    DOUBLE PRECISION SUM_PART, F_FUNCTION

    SUM_PART = 0.d0

    DO i=1,kc
        SUM_PART = SUM_PART + F_FUNCTION(i, aa, m) * (X(1) ** 2 + X(i + 2))
    END DO

END FUNCTION SUM_PART

SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

! Evaluates the algebraic equations or ODE right hand side

! Input arguments :
!      NDIM   :   Dimension of the ODE system 
!      U      :   State variables
!      ICP    :   Array indicating the free parameter(s)
!      PAR    :   Equation parameters

! Values to be returned :
!      F      :   ODE right hand side values

! Normally unused Jacobian arguments : IJAC, DFDU, DFDP (see manual)

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM),DFDP(NDIM,*)

      DOUBLE PRECISION Pi,gamma,aa
      DOUBLE PRECISION SUM_PART, F_FUNCTION, G_FUNCTION, H_FUNCTION
      INTEGER kc, i, m

      Pi=dacos(-1.D0)

  !
  !  FUNDAMENTAL PARAMETERS
  !
      gamma=PAR(1)
      aa=PAR(2)
      m=int(PAR(3))
      kc=NDIM - 2

  !
  !  EVOLUTION EQUATIONS
  !

      F(1) = U(2) - gamma * U(1)
      F(2) = - (gamma / 2) * U(2) + (1 + (gamma**2) / 2) * U(1) - U(1) * SUM_PART(U, kc, aa, m)
      
      DO i=1,kc
          F(i + 2) = gamma * (G_FUNCTION(i, aa) * U(1) ** 2 - H_FUNCTION(i, aa) * U(i+2))
      END DO

!      print*, PAR(1), PAR(2), PAR(3)
!    print*, F

END SUBROUTINE FUNC
!----------------------------------------------------------------------
!----------------------------------------------------------------------

 SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

! Input arguments :
!      NDIM   :   Dimension of the ODE system 

! Values to be returned :
!      U      :   A starting solution vector
!      PAR    :   The corresponding equation-parameter values

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T
      DOUBLE PRECISION :: X(NDIM+1), PARF(9)
      INTEGER :: i,is,kc

      DOUBLE PRECISION Pi

      Pi=dacos(-1.D0)
  

! Initialize the equation parameters
      PAR(1)=0.5 ! gamma
      PAR(2)= Pi*sqrt(2.) ! a
      PAR(3)=1 ! m
      kc = NDIM - 2

! Initialize the solution
!      U(1)=1.d0
!      U(2)=1.d0
!
!      DO i=1,kc
!          U(i+2)=-U(1)**2
!      END DO
     U = 0.D0

 END SUBROUTINE STPNT

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)
!--------- ----

! Boundary Conditions

! Input arguments :
!      NDIM   :   Dimension of the ODE system 
!      PAR    :   Equation parameters
!      ICP    :   Array indicating the free parameter(s)
!      NBC    :   Number of boundary conditions
!      U0     :   State variable values at the left boundary
!      U1     :   State variable values at the right boundary

! Values to be returned :
!      FB     :   The values of the boundary condition functions 

! Normally unused Jacobian arguments : IJAC, DBC (see manual)

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, ICP(*), NBC, IJAC
  DOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)
  DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
  DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)

!X FB(1)=
!X FB(2)=

END SUBROUTINE BCND

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT)
!--------- ----

! Integral Conditions

! Input arguments :
!      NDIM   :   Dimension of the ODE system 
!      PAR    :   Equation parameters
!      ICP    :   Array indicating the free parameter(s)
!      NINT   :   Number of integral conditions
!      U      :   Value of the vector function U at `time' t

! The following input arguments, which are normally not needed,
! correspond to the preceding point on the solution branch
!      UOLD   :   The state vector at 'time' t
!      UDOT   :   Derivative of UOLD with respect to arclength
!      UPOLD  :   Derivative of UOLD with respect to `time'

! Normally unused Jacobian arguments : IJAC, DINT

! Values to be returned :
!      FI     :   The value of the vector integrand 

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, ICP(*), NINT, IJAC
  DOUBLE PRECISION, INTENT(IN) :: PAR(*)
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), UOLD(NDIM), UDOT(NDIM), UPOLD(NDIM)
  DOUBLE PRECISION, INTENT(OUT) :: FI(NINT)
  DOUBLE PRECISION, INTENT(INOUT) :: DINT(NINT,*)

!X FI(1)=

END SUBROUTINE ICND

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE FOPT(NDIM,U,ICP,PAR,IJAC,FS,DFDU,DFDP)
!--------- ----
!
! Defines the objective function for algebraic optimization problems
!
! Supplied variables :
!      NDIM   :   Dimension of the state equation
!      U      :   The state vector
!      ICP    :   Indices of the control parameters
!      PAR    :   The vector of control parameters
!
! Values to be returned :
!      FS      :   The value of the objective function
!
! Normally unused Jacobian argument : IJAC, DFDP

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
  DOUBLE PRECISION, INTENT(OUT) :: FS
  DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM),DFDP(*)

!X FS=

END SUBROUTINE FOPT

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE PVLS(NDIM,U,PAR)
!--------- ----

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NDIM
  DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM)
  DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)
  DOUBLE PRECISION :: GETP,pi,realfm,imagfm,imagfm1
  DOUBLE PRECISION :: lw,lw1
  LOGICAL, SAVE :: first = .FALSE.              
  DOUBLE PRECISION :: T
  INTEGER :: i

  IF (first) THEN
     CALL STPNT(NDIM,U,PAR,T)
     first = .FALSE.
  ENDIF

  !PAR(26)=U(44)
  !PAR(27)=U(52)
  PAR(25)=0.
  pi = 4*ATAN(1d0)
  i=1
  lw=100.
  lw1=101.
  DO WHILE(i < NDIM)
      realfm = GETP('EIG',I*2-1,U)
      IF (ABS(realfm) < lw) THEN
          lw = ABS(realfm)
          lw1 = ABS(GETP('EIG',(I+1)*2-1,U))
          imagfm1 = ABS(GETP('EIG',(I+1)*2,U))
          imagfm = ABS(GETP('EIG',I*2,U))
      END IF
      i=i+1
  END DO
  IF ((lw==lw1).AND.(imagfm1==imagfm).AND.(imagfm/=0.D0)) THEN
    PAR(25) = 2*pi/imagfm
  ENDIF



!---------------------------------------------------------------------- 
! NOTE : 
! Parameters set in this subroutine should be considered as ``solution 
! measures'' and be used for output purposes only.
! 
! They should never be used as `true'' continuation parameters. 
!
! They may, however, be added as ``over-specified parameters'' in the 
! parameter list associated with the AUTO-Constant NICP, in order to 
! print their values on the screen and in the ``p.xxx file.
!
! They may also appear in the list associated with AUTO-Constant NUZR.
!
!---------------------------------------------------------------------- 
! For algebraic problems the argument U is, as usual, the state vector.
! For differential equations the argument U represents the approximate 
! solution on the entire interval [0,1]. In this case its values must 
! be accessed indirectly by calls to GETP, as illustrated below.
!---------------------------------------------------------------------- 
!
! Set PAR(2) equal to the L2-norm of U(1)
!X PAR(2)=GETP('NRM',1,U)
!
! Set PAR(3) equal to the minimum of U(2)
!X PAR(3)=GETP('MIN',2,U)
!
! Set PAR(4) equal to the value of U(2) at the left boundary.
!X PAR(4)=GETP('BV0',2,U)
!
! Set PAR(5) equal to the pseudo-arclength step size used.
!X PAR(5)=GETP('STP',1,U)
!
!---------------------------------------------------------------------- 
! The first argument of GETP may be one of the following:
!        'NRM' (L2-norm),     'MAX' (maximum),
!        'INT' (integral),    'BV0 (left boundary value),
!        'MIN' (minimum),     'BV1' (right boundary value).
!
! Also available are
!   'STP' (Pseudo-arclength step size used).
!   'FLD' (`Fold function', which vanishes at folds).
!   'BIF' (`Bifurcation function', which vanishes at singular points).
!   'HBF' (`Hopf function'; which vanishes at Hopf points).
!   'SPB' ( Function which vanishes at secondary periodic bifurcations).
!---------------------------------------------------------------------- 


END SUBROUTINE PVLS




!----------------------------------------------------------------------
!----------------------------------------------------------------------
