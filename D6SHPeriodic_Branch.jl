#Computer assisted proof of a D₆ branch of periodic solutions for the 2D Swift Hohenberg equation 
# The following code computes the solution and rigorously proves the results given in Theorems 4.7, 4.8, and 4.9 of
# "Proving periodic solutions and branches in the 2D Swift Hohenberg PDE with hexagonal and triangular symmetry"  Dominic Blanco

# We provide the data for the approximate solution, ū. 
# Then, we perform the steps described in Section 4.1 to build w̄(s), B(s), and B^†(s)
# From this we can check if the proof of the solution is verified or not.

#####################################################################################################################################################################

# Needed packages
using RadiiPolynomial, LinearAlgebra, JLD2

# Needed additional sequence structures for RadiiPolynomial
include("dihedral.jl")

#####################################################################################################################################################################


#################################### List of the needed functions : go directly to line 404 for the main code ################################################# 

#The map f defined in Section 3.1. Its derivative is below.
function f(u,μ,γ)
    f1 = similar(u)
    Δ = project(Laplacian(2),space(u),space(u),Float64)
    project!(f1,-(I + Δ)^2*u - μ*u + γ*u^2 - u^3)
    return f1
end

function Df(u,μ,γ)
    Df1 = LinearOperator(space(u),space(u),zeros(dimension(space(u)),dimension(space(u))))
    Δ = project(Laplacian(2),space(u),space(u),Float64)
    U = project(Multiplication(u),space(u),space(u),Float64)
    U² = project(Multiplication(u^2),space(u),space(u),Float64)
    Df1 = -(I + Δ)^2 - μ*I + 2γ * U - 3* U²
    return Df1
end

# A Newton method on the map f.
function _newton(a,μ,γ)
    ϵ = 1
    nv = 1
    j = 0
    while (ϵ > 1e-14) & (j < 15)
        F = f(a,μ,γ)
        DF = Df(a,μ,γ)
        a = a - DF\F
        @show ϵ = norm(F,Inf)
        if ϵ > 7
            display("Newton may have diverged")
            CrashNow = Badk
            return a,ϵ
        end
        nv = norm(a)
        if nv < 1e-5
            @show nv = norm(a)
            display("Newton may have converged to the 0 solution")
            return nv,j
            break
        end
        j += 1
    end
    return a,ϵ
end

# The map F defined in Section 4. Its derivative is below.
function F(X,ū,u̇,γ)
    fc1 = similar(X)
    μ,u = eachcomponent(X)
    component(fc1,1)[1] = sum((u - ū).*u̇)
    Δ = project(Laplacian(2),space(u),space(u),Float64)
    project!(component(fc1,2),-(I + Δ)^2*u - μ[1]*u + γ*u^2 - u^3)
    return fc1
end

function DF(X,u̇,γ)
    Dfc1 = LinearOperator(space(X),space(X), zeros(dimension(space(X)),dimension(space(X))))
    μ,u = eachcomponent(X)
    Dfc1 .= 0
    component(Dfc1,1,2)[1,:] .= u̇
    project!(component(Dfc1,2,1),project(-u,space(u)))
    Δ = project(Laplacian(2),space(u),codomain(Laplacian(2),space(u)),Float64)
    U = project(Multiplication(u),space(u),space(u),Float64)
    U² = project(Multiplication(u^2),space(u),space(u),Float64)
    project!(component(Dfc1,2,2),-(I + Δ)^2 - μ[1]*I + 2γ * U - 3* U²)
    return Dfc1
end

# Computes the tangent vector to the current point on the branch.
function TangentVector(X,Ẋ,γ)
    μ,u = eachcomponent(X)
    u̇ = project(-u,space(u))
    Δ = project(Laplacian(2),space(u),codomain(Laplacian(2),space(u)),Float64)
    U = project(Multiplication(u),space(u),space(u),Float64)
    U² = project(Multiplication(u^2),space(u),space(u),Float64)
    DF = -(I + Δ)^2 - μ[1]*I + 2γ * U - 3* U²
    DF⁻¹ = inv(DF)
    u̇ = DF⁻¹*u̇
    Ẋn = zero(Ẋ)
    component(Ẋn,1)[1] = -1
    component(Ẋn,2) .= u̇
    Ẋn = Ẋn/norm(Ẋn)
    @show Σ = sum(component(Ẋn,2).*component(Ẋ,2))
    if (Σ > 0)
        return Ẋn
    else
        return -Ẋn
    end
end

# Performs the numerical continuation using Newton.
function _newton_continue(ū,μ,Pfourier,arclength_grid,γ,Ẋ,npts)
    X = Sequence(Pfourier, [μ ; ū[:]])
    norm_u_grid = []
    μ_grid = []
    Xvec = []
    Ẋvec = []
    push!(Xvec,X)
    push!(Ẋvec,Ẋ)
    push!(μ_grid,μ)
    push!(norm_u_grid, norm(ū,2))
    μloop = μ
    for k = 2:npts
        Ẋ = TangentVector(X,Ẋ,γ)
        dμ = arclength_grid[k] - arclength_grid[k-1]
        X = X + dμ*Ẋ
        ū = component(X,2)
        Y,ϵ = _newton_s(X,Ẋ,ū,γ)
        if ϵ > 7
            return norm_u_grid,μ_grid,Xvec,Ẋvec,X,Ẋ
        elseif ϵ == 15
            return norm_u_grid,μ_grid,Xvec,Ẋvec,X,Ẋ
        else
            X = Y
        end
        J = component(X,2)
        push!(norm_u_grid,norm(J,2))
        push!(μ_grid,component(X,1)[1])
        μloop = component(X,1)[1]
        @show μloop
        d = π/frequency(J)[1]
        push!(Xvec,X)
        push!(Ẋvec,Ẋ)
        @show k
    end
    return norm_u_grid,μ_grid,Xvec,Ẋvec
end

# Runs a Newton method on the map F. 
function _newton_s(X,Ẋ,ū,γ)
    ϵ = 1
    j = 0
    u̇ = component(Ẋ,2)
    while (ϵ > 1e-14) & (j < 15)
    FC = F(X,ū,u̇,γ)
    DFC = DF(X,u̇,γ)
    X = X - DFC\FC
    @show ϵ = norm(FC,Inf)
    nv = norm(X,Inf)
    if nv < 1e-5
        print("Newton may have converged to the zero solution")
    end
    if ϵ > 7
        @show opnorm(inv(DFC),2)
        print("Newton may have diverged")
        return zero(X),ϵ
    end
    if (Int(j + 1) == 15)
        @show opnorm(inv(DFC),2)
        print("Newton is having trouble converging")
        return zero(X),15
    end
    j += 1
    end
    return X,ϵ
end

# Computes the norm on Chebysev series of the form described in Section 4.
function norm_cheb(s,X)
    norm_ans = norm(s[1],X)
    l = length(s)
    for i = 2:l 
        norm_ans += interval(2)*norm(s[i],X)
    end 
    return norm_ans
end

# Allows us to switch between D₆ and exponential Fourier series.
function _build_P(ν,space)
    ord = order(space)[1]
    V = interval.(vec(zeros(dimension(space))))
    V[1] = interval(1)
    for k₁ = 1:div(ord,2)
        V[2k₁ + k₁*div(ord,2) - div((k₁-1)^2 + 3*(k₁-1),2)] = interval(4)*ν^(interval(3k₁)) + interval(2)*ν^(interval(2k₁))
        V[k₁ + 1] = interval(2)*ν^(interval(2k₁)) + interval(4)*ν^(interval(k₁))
    end
    for k₂ = 1:div(ord,2)
        for k₁ = (2k₂+1):(k₂ + div(ord,2))
            V[k₁ + k₂*div(ord,2) - div((k₂-1)^2 + 3*(k₂-1),2)] = interval(4)*(ν^(interval(k₁+k₂)) + ν^(interval(k₂ + abs(k₁ - k₂))) + ν^(interval(k₁ + abs(k₁ - k₂))))
        end
    end
    return V
end

# Computes the operator norm on Chebyshev series as defined in Section 4.
function opnorm_cheb(s,ν)
    dom = domain(s[1])[2]
    codom = codomain(s[1])[2]
    P = _build_P(ν,codom)
    P⁻¹ = interval.(ones(dimension(dom)))./_build_P(ν,dom)
    P = [interval(1) ; P]
    P⁻¹ = [interval(1) ; P⁻¹]
    opnorm_ans = opnorm(P.*s[1].*P⁻¹',1)
    l = length(s)
    for i = 2:l 
        opnorm_ans += interval(2)*opnorm(P.*s[i].*P⁻¹',1)
    end 
    return opnorm_ans
end

# Takes a grid of ParameterSpace() × D₆Fourier sequences and fits a Chebyshev sequence to it.
grid2cheb(x_fft, N) = 
    [rifft!(complex.(getindex.(x_fft, i)), Chebyshev(N)) for i ∈ indices(space(x_fft[1]))]

# Takes a grid of ParameterSpace() × D₆Fourier operators and fits a Chebyshev sequence to it.
grid2chebm(x_fft, N) =
    [rifft!(complex.(getindex.(x_fft, i, j)), Chebyshev(N)) for i ∈ indices(codomain(x_fft[1])), j ∈ indices(domain(x_fft[1]))]

# Takes a Chebyshev sequence and converts it back to a grid of ParameterSpace() × D₆Fourier sequences or operators.
function cheb2grid(x::VecOrMat{<:Sequence}, N_fft)
    vals = RadiiPolynomial.fft.(x, N_fft)
    return [real.(getindex.(vals, i)) for i ∈ eachindex(vals[1])]
end

#The map F with intervals for rigorous computations.
function Fi(X,ū,u̇,γ)
    fic1 = similar(X)
    μ,u = eachcomponent(X)
    component(fic1,1)[1] = sum((u - ū).*u̇)
    N = order(space(u))[1]
    f = frequency(space(u))[1]
    Δ = project(Laplacian(2),D₆Fourier(N,mid(f)),D₆Fourier(N,mid(f)),Float64)
    L_diag = interval(1) .+ interval.(diag(coefficients(Δ)))
    L = LinearOperator(space(u),space(u), Diagonal(L_diag.^2))
    project!(component(fic1,2),-L*u - μ[1]*u + γ*u^2 - u^3)
    return fic1
end

# Evaluates Fi component-wise.
function F_V(X,Ẋ,γ,pfourier)
    l = length(X)
    F1 = Vector{Sequence{CartesianProduct{Tuple{ParameterSpace, D₆Fourier}}, Vector{Interval{Float64}}}}(undef,l)
    for i = 1:l
        Xi = Sequence(pfourier, X[i]) 
        Ẋi = Sequence(pfourier, Ẋ[i])
        Vi = component(Xi,2)
        V̇i = component(Ẋi,2)
        F1[i] = Fi(Xi,Vi,V̇i,γ)
    end

    return F1
end

#Converts a vector of vectors to a vector of sequences.
function _vec_to_seq(x,N,pfourier)
    l = length(x)
    y = Vector{Sequence{CartesianProduct{Tuple{ParameterSpace, D₆Fourier}},Vector{Interval{Float64}}}}(undef,N)
    for k = 1:l 
        y[k] = Sequence(pfourier, x[k])
    end
    return y 
end

# Computes the tail of G, the nonlinear part for the Y₀ˢ bound.
function _compute_tail(X_fft,γ,pfourier)
    l = length(X_fft)
    G_tail = Vector{Sequence{D₆Fourier,Vector{Interval{Float64}}}}(undef,l)
    for i = 1:l 
        Xi = Sequence(pfourier,X_fft[i])
        Vi = component(Xi,2)
        G_tail[i] = γ*Vi^2 - Vi^3 - project(γ*Vi^2 - Vi^3,pfourier[2])
    end
    return G_tail
end

#Converts a grid to a Chebyshev sequences only for D₆Fourier.
grid2chebonlyD6(x_fft, N) = 
    [rifft!(complex.(getindex.(x_fft, TensorIndices(i))), Chebyshev(N)) for i ∈ indices(space(x_fft[1]))]


# The following functions perform what their names indicate.
function _A_as_a_cheb_sequence_of_D6fourier_operators(A_cheb,Nc,pfourier)
    A_cheb0mat =  [real.(getindex.(A_cheb, i)) for i ∈ eachindex(coefficients(A_cheb[1])).-1] 
    A_cheb0 = Vector{LinearOperator{CartesianProduct{Tuple{ParameterSpace, D₆Fourier}}, CartesianProduct{Tuple{ParameterSpace, D₆Fourier}}, Matrix{Interval{Float64}}}}(undef,Nc+1)
    for i = 1:Nc+1 
        A_cheb0[i] = LinearOperator(pfourier,pfourier, A_cheb0mat[i])
    end
    return A_cheb0 
end

function _X_as_a_cheb_sequence_of_D6fourier_sequences(X_cheb,Nc,pfourier)
    X_chebvec = [real.(getindex.(X_cheb, i)) for i ∈ eachindex(coefficients(X_cheb[1])).-1]
    X_cheb0 = Vector{Sequence{CartesianProduct{Tuple{ParameterSpace, D₆Fourier}},Vector{Interval{Float64}}}}(undef,Nc+1)
    for i = 1:Nc+1
        X_cheb0[i] = Sequence(pfourier, X_chebvec[i])
    end
    return X_cheb0 
end

# Computes V and ϕ for the Z₁ˢ bound.
function _compute_v_ϕ(X_fft,γ,pfourier,ν)
    l = length(X_fft)
    N = order(pfourier[2])[1]
    f = frequency(pfourier[2])[1]
    fourier2 = D₆Fourier(2N,f)
    dim = dimension(fourier)
    v = Vector{Sequence{D₆Fourier,Vector{Interval{Float64}}}}(undef,l)
    ϕ = Vector{Sequence{CartesianProduct{Tuple{ParameterSpace, D₆Fourier}},Vector{Interval{Float64}}}}(undef,l)
    for i = 1:l 
        Xi = Sequence(pfourier,X_fft[i])
        Vi = component(Xi,2)
        v[i] = interval(2)*γ*Vi - interval(3)*Vi^2
        ϕ[i] = Sequence(pfourier, [interval(0) ; interval.(ones(dim))*norm(Sequence(fourier2, [interval(0) ; coefficients(v[i])[2:end]]),Inf)/ν^(interval(N+1))])
    end
    return v,ϕ
end

# Converts a matrix to a linear operator.
function _mat_to_linop(M,pfourier)
    N = order(pfourier[2])[1]
    l = length(M)
    Mop = Vector{LinearOperator{CartesianProduct{Tuple{ParameterSpace, D₆Fourier}},CartesianProduct{Tuple{ParameterSpace, D₆Fourier}},Matrix{Interval{Float64}}}}(undef,l)
    for i = 1:l 
        Mop[i] = LinearOperator(pfourier,pfourier, M[i])
    end
    return Mop 
end

# Subtracts a linear operator from the identity.
function _subtract_from_identity(Mop)
    l = length(Mop)
    Msub = Vector{LinearOperator{CartesianProduct{Tuple{ParameterSpace, D₆Fourier}},CartesianProduct{Tuple{ParameterSpace, D₆Fourier}},Matrix{Interval{Float64}}}}(undef,l)
    for i = 1:l 
        Msub[i] = UniformScaling(interval(1)) - Mop[i] 
    end
    return Msub 
end

function I_minus_ADF_as_a_cheb_sequence_of_D6fourier_operators(I_minus_ADF_cheb,N3,pfourier)
    I_minus_ADF_cheb_0mat =  [real.(getindex.(I_minus_ADF_cheb, i)) for i ∈ eachindex(coefficients(I_minus_ADF_cheb[1])).-1] 
    I_minus_ADF_cheb0 = Vector{LinearOperator{CartesianProduct{Tuple{ParameterSpace, D₆Fourier}}, CartesianProduct{Tuple{ParameterSpace, D₆Fourier}}, Matrix{Interval{Float64}}}}(undef,N3+1)
    N = order(pfourier[2])[1]
    for i = 1:N3+1 
        I_minus_ADF_cheb0[i] = LinearOperator(pfourier,pfourier, I_minus_ADF_cheb_0mat[i])
    end
    return I_minus_ADF_cheb0 
end

# The derivative DF on intervals.
function DFi(X,u̇,γ)
    Dfc1 = LinearOperator(space(X),space(X), interval.(zeros(dimension(space(X)),dimension(space(X)))))
    μ,u = eachcomponent(X)
    Dfc1 .= interval(0)
    component(Dfc1,1,2)[1,:] .= u̇
    project!(component(Dfc1,2,1),project(-u,space(u)))
    N = order(space(u))[1]
    f = frequency(space(u))[1]
    Δ = project(Laplacian(2),D₆Fourier(N,mid(f)),D₆Fourier(N,mid(f)),Float64)
    L_diag = interval(1) .+ interval.(diag(coefficients(Δ)))
    L = LinearOperator(space(u),space(u), Diagonal(L_diag.^2))
    U = project(Multiplication(u),space(u),space(u),Interval{Float64})
    U² = project(Multiplication(u^2),space(u),space(u),Interval{Float64})
    project!(component(Dfc1,2,2),-L - μ[1]*UniformScaling(interval(1)) + interval(2)*γ * U - interval(3)* U²)
    return Dfc1
end

# Evaluates DFi component-wise.
function DF_V(X,Ẋ,γ,pfourier)
    l = length(X)
    DF1 = Vector{LinearOperator{CartesianProduct{Tuple{ParameterSpace, D₆Fourier}}, CartesianProduct{Tuple{ParameterSpace, D₆Fourier}}, Matrix{Interval{Float64}}}}(undef,l)
    for i = 1:l 
        Xi = X[i]
        Ẋi = Ẋ[i]
        V̇i = component(Ẋi,2)
        DF1[i] = DFi(Xi,V̇i,γ)
    end
    return DF1
end

# Checks the conditions of the Radii-Polynomial Theorem (see Section 4).
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

######################################### Main Code ###################################
# Branch 1
ū = load("ubar_Th_4_7","ubar")
N = 60
μ = interval(0.01)
γ = interval(1.6)
d = interval(10) 
ν = interval(1.1)
r₀ = interval(1e-4)
# Note that the arclength is not truly negative. This is us choosing the direction we are continuing. This is the simplest way to do so numerically.
arclength = -0.4
Nc = 7
#= Branch 2
ū = load("ubar_Th_4_8","ubar")
N = 20
μ = interval(-0.01)
γ = interval(1.6)
d = interval(5)
ν = interval(1.25)
r₀ = interval(5e-5)
arclength = 0.18
Nc = 31=#
#= Branch 3
ū = load("ubar_Th_4_9","ubar")
N = 20
μ = interval(-0.175)
γ = interval(1.6)
d = interval(5)
ν = interval(1.25)
r₀ = interval(3e-5)
arclength = -0.23
Nc = 15=#

fourier = D₆Fourier(N,π/d)
fourier_mid = D₆Fourier(N,mid(π/d))
ℓ¹ν = Ell1(GeometricWeight(ν))
X = NormedCartesianSpace((ℓ¹(),ℓ¹ν),ℓ¹())
pfourier = ParameterSpace() × fourier

# Building the objects of Section 4.1.
N_fft = nextpow(2, 2Nc + 1)
npts = N_fft ÷ 2 + 1

arclength_grid = [0.5 * arclength - 0.5 * cospi(2j/N_fft) * arclength for j ∈ 0:npts-1]

pfourier_mid = ParameterSpace() × fourier_mid
ẇ = TangentVector(Sequence(pfourier_mid, [mid(μ) ; coefficients(ū)]),Sequence(pfourier_mid, zeros(dimension(pfourier_mid))),mid(γ))
norm_u_grid,μ_grid,w_grid,ẇ_grid = _newton_continue(ū,mid(μ),pfourier_mid,arclength_grid,mid(γ),ẇ,npts)
L_NK = abs((interval(1) + (sqrt(interval(3))/interval(2) * (interval(N+1))*π/d)^2)^2 + interval(findmin(μ_grid)[1]))


w_fft = [w_grid ; reverse(w_grid)[2:npts-1]]
ẇ_fft = [ẇ_grid ; reverse(ẇ_grid)[2:npts-1]]
w̄_cheb = interval.(grid2cheb(w_fft,Nc))
ẇ_cheb = interval.(grid2cheb(ẇ_fft,Nc))

B_grid = inv.(DF.(w_fft,component.(ẇ_fft,2),mid(γ)))
B_cheb = interval.(grid2chebm(B_grid, Nc))
# We now begin the computer assisted proof.
################################## The Bound Y₀ˢ ############################
# The bound Y₀ˢ as in Lemma 4.2.
# B(s)F(w̄(s)) is a polynomial with respect to s of order 4N
N4 = 4Nc
N4_fft = nextpow(2, 2N4 + 1)
npts_N4 = N4_fft ÷ 2 + 1

w̄_fft_N4 = cheb2grid(w̄_cheb,N4_fft)
ẇ_fft_N4 = cheb2grid(ẇ_cheb,N4_fft)
Ff = F_V(w̄_fft_N4,ẇ_fft_N4,γ,pfourier)

BF_fft = cheb2grid(B_cheb,N4_fft).*coefficients.(Ff)

BF_cheb = grid2cheb(complex.(_vec_to_seq(BF_fft,N4_fft,pfourier)), N4)
BF_cheb = [real.(getindex.(BF_cheb, i)) for i ∈ eachindex(coefficients(BF_cheb[1])).-1]
BF_cheb = [Sequence(pfourier,BF_cheb[i]) for i = 1:length(BF_cheb)]

#Tail part
N3 = 3Nc
N3_fft = nextpow(2, 3N3 + 1)
npts_N3 = N3_fft ÷ 2 + 1
    
w̄_fft_N3 = cheb2grid(w̄_cheb,N3_fft)

G_tail_fft = _compute_tail(w̄_fft_N3,γ,pfourier)

G_tail_cheb = grid2chebonlyD6(complex.(G_tail_fft), N3)
G_tail_cheb = [real.(getindex.(G_tail_cheb, i)) for i ∈ eachindex(coefficients(G_tail_cheb[1])).-1]
G_tail_cheb = [Sequence(D₆Fourier(3N,π/d),G_tail_cheb[i]) for i = 1:length(G_tail_cheb)]

Y₀ˢ = norm_cheb(BF_cheb,X) + interval(2Nc+1)/L_NK*norm_cheb(G_tail_cheb,ℓ¹ν)
@show Y₀ˢ
############################## The Bound Z₂ˢ ############################
# The bound Z₂ˢ as in Lemma 4.3.
B_cheb0 = _A_as_a_cheb_sequence_of_D6fourier_operators(B_cheb,Nc,pfourier)
w̄_cheb0 = _X_as_a_cheb_sequence_of_D6fourier_sequences(w̄_cheb,Nc,pfourier)
norm_B_cheb0 = opnorm_cheb(B_cheb0,ν)
fourier2 = D₆Fourier(2N,π/d)

q_cheb0 = interval(2)*γ .- interval(6)*component.(w̄_cheb0,2)
Z₂ˢ = (norm_B_cheb0 + interval(2Nc+1)/L_NK)*(interval(1) + norm_cheb(q_cheb0,ℓ¹ν) + interval(3)*r₀)
@show Z₂ˢ
############################ The Bound Z₁ˢ ##########################
# The bound Z₁ˢ as in Lemma 4.4.
# B(s)(0,ϕ(s)) is a polynomial with respect to s of order 2N
N2 = 2Nc
N2_fft = nextpow(2, 2N2 + 1)
npts_N2 = N2_fft ÷ 2 + 1
w̄_fft_N2 = cheb2grid(w̄_cheb,N2_fft)
v̄_fft,ϕ_fft = _compute_v_ϕ(w̄_fft_N2,γ,pfourier,ν)
v̄_cheb = grid2chebonlyD6(v̄_fft,N2)
v̄_cheb = [real.(getindex.(v̄_cheb, i)) for i ∈ eachindex(coefficients(v̄_cheb[1])).-1]
v̄_cheb = [Sequence(D₆Fourier(2N,π/d),v̄_cheb[i]) for i = 1:length(v̄_cheb)]
Bϕ_fft_N2 = _vec_to_seq(cheb2grid(B_cheb,N2_fft).*coefficients.(ϕ_fft),N2_fft,pfourier)
Bϕ_cheb_N2 = grid2cheb(Bϕ_fft_N2,N2)
Bϕ_cheb_N2 = [real.(getindex.(Bϕ_cheb_N2, i)) for i ∈ eachindex(coefficients(Bϕ_cheb_N2[1])).-1]
Bϕ_cheb_N2 = [Sequence(pfourier,Bϕ_cheb_N2[i]) for i = 1:length(Bϕ_cheb_N2)]
Z₁ˢ = norm_cheb(Bϕ_cheb_N2,X) + interval(2Nc+1)/L_NK*norm_cheb(v̄_cheb,ℓ¹ν)
@show Z₁ˢ
############################ The Bound Z₀ˢ ##########################
# The bound Z₀ˢ as in Lemma 4.2.
# B(s)B(s)^† is a polynomial with respect to s of order 2N
w̄_fft_N2 = _vec_to_seq(w̄_fft_N2,N2_fft,pfourier)
ẇ_fft_N2 = _vec_to_seq(cheb2grid(ẇ_cheb,N2_fft),N2_fft,pfourier)

DFf = DFi.(w̄_fft_N2,component.(ẇ_fft_N2,2),γ)
BDF_fft = _mat_to_linop(cheb2grid(B_cheb,N2_fft).*coefficients.(DFf),pfourier)
I_minus_BDF_fft = _subtract_from_identity(BDF_fft)
I_minus_BDF_cheb = grid2chebm(I_minus_BDF_fft,N2)
I_minus_BDF_cheb0 = I_minus_ADF_as_a_cheb_sequence_of_D6fourier_operators(I_minus_BDF_cheb,N2,pfourier)
@show Z₀ˢ = opnorm_cheb(I_minus_BDF_cheb0,ν)

#Perform the Computer Assisted Proof of the Branch
r_min = sup((interval(1) - Z₁ˢ - Z₀ˢ - sqrt((interval(1) - Z₁ˢ-Z₀ˢ)^2 - interval(2)*Y₀ˢ*Z₂ˢ))/Z₂ˢ)
r_max = inf((interval(1) - Z₁ˢ - Z₀ˢ + sqrt((interval(1) - Z₁ˢ-Z₀ˢ)^2 - interval(2)*Y₀ˢ*Z₂ˢ))/Z₂ˢ)
CAP(sup(Y₀ˢ),sup(Z₁ˢ+Z₀ˢ),sup(Z₂ˢ))