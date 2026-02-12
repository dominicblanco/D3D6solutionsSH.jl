#Computer assisted proof of a D₃ periodic solution for the 2D Swift Hohenberg equation 
# The following code computes the solution and rigorously proves the results given in section 3.4 of
# "Proving periodic solutions and branches in the 2D Swift Hohenberg PDE with hexagonal and triangular symmetry"  Dominic Blanco

# We provide the data for the approximate solution, ū. 
# From this we can check if the proof of the solution is verified or not.

#####################################################################################################################################################################

# Needed packages
using RadiiPolynomial, LinearAlgebra, JLD2

# Needed additional sequence structures for RadiiPolynomial
include("dihedral.jl")

#####################################################################################################################################################################


#################################### List of the needed functions : go directly to line 52 for the main code ################################################# 

# Allows us to switch between D₃ and exponential Fourier series
function _build_P(ν,space)
    ord = order(space)[1]
    V = vec(interval.(zeros(dimension(space))))
    V[1] = interval(1)
    for k₁ = 1:div(ord,2)
        V[k₁ + k₁*(div(ord,2)+1) - (k₁-1)] = ν^(interval(2k₁)) + interval(2)*ν^(interval(k₁))
        V[k₁ + 1] = ν^(interval(2k₁)) + interval(2)*ν^(interval(k₁))
    end
    for k₂ = 1:div(ord,2)
        for k₁ = (k₂+1):(k₂ + div(ord,2))
            V[k₁ + k₂*(div(ord,2)+1) - (k₂-1)] = interval(2)*(ν^(interval(k₁+k₂)) + ν^(interval(k₂ + abs(k₁ - k₂))) + ν^(interval(k₁ + abs(k₁ - k₂))))
        end
    end
    return V
end

# Checks the conditions of the Radii-Polynomial Theorem (see Theorem 3.1).
function CAP(Y₀,Z₁,Z₂)
    if Z₁ > 1
        display("Z₁ is too big")
        return Z₁
    elseif 2Y₀*Z₂ > (1-Z₁)^2
        display("The condition 2Y₀*Z₂ < (1-Z₁)² is not satisfied")
        return Y₀,Z₁,Z₂
    else
        display("The computer assisted proof was successful!")
        return Y₀,Z₁,Z₂
    end
end

################### PROOF OF D₃ SOLUTION : MAIN CODE #################################################################################################################################################
#Solution 1
N = 70
d = interval(10)
μ = interval(0.01)
γ = interval(1.6)
r₀ = interval(3e-5)
ū = load("ubar_Th_3_5","ubar")
ν = interval(1.15)
#= Solution 2
N = 12
d = interval(5)
μ = interval(0.01)
γ = interval(1.6)
r₀ = interval(2e-4)
ū = load("ubar_Th_3_6","ubar")
ν = interval(1.38)=#
#=Solution 3
N = 14
d = interval(5)
μ = interval(-0.01)
γ = interval(1.7)
r₀ = interval(2e-4)
ū = load("ubar_Th_3_7","ubar")
ν = interval(1.34)=#
#=Solution 4
N = 10
d = interval(5)
μ = interval(-0.2)
γ = interval(2)
r₀ = interval(9e-4)
ū = load("ubar_Th_3_8","ubar")
ν = interval(1.33)=#

fourier = D₃Fourier(N,π/d)
ū_interval = Sequence(fourier, interval.(coefficients(ū)))

L = -(UniformScaling(interval(1)) + LinearOperator(fourier,fourier,coefficients(interval.(project(Laplacian(2), D₃Fourier(N,mid.(π/d)), D₃Fourier(N,mid.(π/d)),Float64)))))^2 - μ*UniformScaling(interval(1))
L⁻¹ = interval.(ones(dimension(fourier)))./L

X = Ell1(GeometricWeight(ν))
# # We define an operator P that help us to switch between the D₃ and exponential series
# # (as the theoretical analysis is done in exponential series)
# # For a linear operator B between D₃ fourier series, P*B*inv(P) gives the equivalent operator
# # on exponential series for the D₃ modes (the other modes can be found by computing the orbits of the stored modes)
# # In particular, if B is diagonal, then P*B*inv(P) = B
P = _build_P(ν,fourier)
P⁻¹ = interval.(ones(dimension(fourier)))./P
P⁻¹2 = interval.(ones(dimension(D₃Fourier(2N,π/d))))./_build_P(ν,D₃Fourier(2N,π/d))
# Computation of A and its norm
ū²_interval = ū_interval^2
v̄_interval = interval(2)*γ*ū_interval - interval(3)*ū²_interval
𝕧̄ = project(Multiplication(v̄_interval),fourier,fourier,Complex{Interval{Float64}})
A = interval.(inv(mid.(L + 𝕧̄)))
norm_A = opnorm(LinearOperator(coefficients(P.*A.*P⁻¹')),1)
L_N = abs((interval(1) + (sqrt(interval(3))/interval(2) * (interval(N+1))*π/d)^2)^2 + μ)
@show norm_A
################ Y₀ BOUND ######################################################
# Computation of the 𝒴₀ bound, defined in Lemma 3.2.
L_diag = -diag(coefficients(UniformScaling(interval(1)) + interval.(project(Laplacian(2),D₃Fourier(N,mid(π/d)),D₃Fourier(N,mid(π/d)),Float64)))).^2 .- μ
tail_G = γ*ū²_interval - ū²_interval*ū_interval
G = project(tail_G,fourier)
Y₀ = norm(A*project(L_diag.*ū_interval+G,fourier),X) + interval(1)/L_N*norm(tail_G-G,X)
@show Y₀
################################ Z₂ BOUND ######################################################
# Computation of the Z₂ bound defined in Lemma 3.3.
q = interval(2)*γ - interval(6)*ū_interval
Z₂ = (norm_A + interval(1)/L_N)*(norm(q,X) + r₀)
@show Z₂
################################ Z₀ BOUND ######################################################
# Computation of the Z₀ bound defined in Lemma 3.2.
Z₀ = opnorm(LinearOperator(coefficients(P.*(UniformScaling(interval(1)) - A*(L + 𝕧̄)).*P⁻¹')),1)
@show Z₀
################################ Z₁ BOUND ######################################################
# Computation of the Z₁ bound defined in Lemma 3.4.
ϕ = Sequence(fourier, norm(Sequence(D₃Fourier(2N,π/d), [interval(0) ; coefficients(v̄_interval)[2:end]]),Inf)/ν^(interval(N+1))*interval.(ones(dimension(fourier))))
Z₁ = norm(A*ϕ,X) + interval(1)/L_N * norm(v̄_interval,X)
@show Z₁
#Perform the Computer Assisted Proof of the Pattern
r_min = sup((interval(1) - Z₁ - sqrt((interval(1) - Z₁)^2 - interval(2)*Y₀*Z₂))/Z₂)
r_max = inf((interval(1) - Z₁ + sqrt((interval(1) - Z₁)^2 - interval(2)*Y₀*Z₂))/Z₂)
CAP(sup(Y₀),sup(Z₁),sup(Z₂))