

# Set up packages (must have pre-loaded)
using CSV, DataFrames, TableView, Random, Distributions, LinearAlgebra,
      LatexPrint, StatsBase, Plots, SpecialFunctions
using Optim, ForwardDiff

# Set seed
Random.seed!(12345);

# Set plotting backend
# Pkg.add("GR")
gr()

# Read in data
data = CSV.read("psetOne.csv", DataFrame);
showtable(data)
names(data)




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 8. Market t=17
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Prep data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# Select data for market t=17
d17 = data[data.Market .== 17, :];
showtable(d17)

# Outside good share and difference of
s_0 = 1 - sum(d17.shares)
diff_log_s = log.(d17.shares) .- Ref(log(s_0))



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Estimate α
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# Data vectors
Z = Matrix(d17[:, [:z1, :z2, :z3, :z4]])
X = hcat(d17.Constant, -1.0*d17.Price, d17.EngineSize, d17.SportsBike)

# Define objective function
function simple_obj(θ)
      ξ = diff_log_s - X * θ
      outer = Z' * ξ
      W = inv(Z' * Z)
      return outer' * W * outer
end

# Define gradient of objective function
function simple_obj_g(G, θ)
      ξ = diff_log_s - X * θ
      W = inv(Z' * Z)
      grad = 2 * (Z' * (-X))' * W * Z' * ξ
      G[1:length(θ)] = grad[1:length(θ)]
end

# Define initial starting value
θ_initial = ones(4)

# Solve for parameters values
θ_simple = Optim.minimizer(optimize(simple_obj, simple_obj_g,
            θ_initial, BFGS()))
print(θ_simple)

# Save price coefficient
α = θ_simple[2]

# Get ξ from this
ξ_simple = diff_log_s - X * θ_simple


# Check parameters with with TSLS
inv(Z' * X) * (Z' * diff_log_s)

diff_log_s = log.(data.shares) .- Ref(log(s_0));
Z = Matrix(d17[:, [:z1, :z2, :z3, :z4]]);
X = hcat(d17.Constant, -1.0*d17.Price, d17.EngineSize, d17.SportsBike);
inv(Z' * X) * (Z' * diff_log_s)






# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Solve p = - (Δ * J_p)^{-1} D_{.t}
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# Coefficient order: Price, Constant, Engine CC, BikeType, Brand2, Brand3
# Data vectors
Z = Matrix(d17[:, [:z1, :z2, :z3, :z4]])
X = hcat(d17[:, [:Price, :Constant, :EngineSize, :SportsBike, :Brand2, :Brand3]])
θ_given =  [-3.0 1.0 1.0 2.0 -1.0 1.0]'
α = θ_given[1]

# Ownership matrix
d17.Brand1 = Ref(1.0) .- max.(d17.Brand2, d17.Brand3);
Δ = (d17.Brand1 .* d17.Brand1') .+ (d17.Brand2 .* d17.Brand2') .+
      (d17.Brand3 .* d17.Brand3')





# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Solve fixed point equation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# Parameter values
θ_given =  [-3.0 1.0 1.0 2.0 -1.0 1.0]'
α = θ_given[1]

# Ownership matrix
d17.Brand1 = Ref(1.0) .- max.(d17.Brand2, d17.Brand3);
Δ = (d17.Brand1 .* d17.Brand1') .+ (d17.Brand2 .* d17.Brand2') .+
      (d17.Brand3 .* d17.Brand3')

# ξ values
ξ_rand = rand(Normal(0,1),7);
ξ = ξ_rand # ξ_simple
ξ = zeros(7)

# With Newton's method
diff = 1;
p_init = d17.Price;
p_init = ones(7);
while diff > 10e-6
      # Compute shares with price
      s = share_newton(p_init)

      # Use computed shares to predict prices
      J_p = J_p_newton(p_init)

      # Compute Ω
      Ω = Δ .* J_p

      # Compute f and ∇f
      func = Ω * p_init - s
      # term1 = Matrix(0.0 * I, 7, 7)
      # for r in 1:7
      #       term1[r,:] = (Δ .* Hess[r,:,:]) * p_init
      # end
      # ∇f = term1 + Ω + J_p
      ∇f = ForwardDiff.jacobian(f_newton, p_init)

      # Newton's method step
      println(diff)
      p_next = p_init - inv(∇f) * func
      diff = abs(maximum(p_next - p_init))
      p_init = copy(p_next)
end
p_next
diff



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

function share_newton(price)
      X = hcat(price, d17.Constant, d17.EngineSize, d17.SportsBike,
            d17.Brand2, d17.Brand3)
      num = exp.(X * θ_given .+ ξ)
      denom = 1 + sum(num)
      s = num / denom
      return s
end

function J_p_newton(price)
      s = share_newton(price)
      # α is negative in this setting
      own_price = α .* diagm(vec(s .* (1 .- s)));
      cross_price = - α * (s .* s') .* (Ref(1) .- Matrix(1.0I, 7, 7));
      J_p = own_price .+ cross_price;
      return J_p
end

function f_newton(price)
      s = share_newton(price)
      J_p = J_p_newton(price)
      Ω = Δ .* J_p
      f = Ω * p_init + s
      return f
end

function Hess_newton(price)
      s = share_newton(price)
      Hess = Array{Float64}(undef, 7, 7, 7)
      for j in 1:7
       for k in 1:7
        for l in 1:7
            Hess[j,k,l] = α^2 *
                  ((2*s[j]*s[k]*s[l] - 3*s[j]*s[k] + s[j])*(j==k && j==l) +
                  (2*s[j]*s[k]*s[l] - s[j]*s[l])*(j==k && j!=l) +
                  (2*s[j]*s[k]*s[l] - s[k]*s[l])*(j!=k && j==l) +
                  (2*s[j]*s[k]*s[l] - s[j]*s[k])*(j!=k && k==l) +
                  (2*s[j]*s[k]*s[l])*(l!=k && k!=j && j!=l))
        end
       end
      end
      return Hess
end


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

x_test = hcat(p_init, d17.Constant, d17.EngineSize, d17.SportsBike,
      d17.Brand2, d17.Brand3)
s = share_newton(p_init)

# Check J_p code
J_p = J_p_newton(p_init)
ForwardDiff.jacobian(share_newton, p_init)

# Check Hess_p code
Hess = Hess_newton(p_init)
Hess_2 = reshape(ForwardDiff.jacobian(J_p_newton, p_init),
            7, 7, 7)


Hess[:,:,1] * p_init

Ω = Δ .* J_p
∇f = ΔHess_p + Ω + J_p

(Δ .* Hess[:,:,1]) * p_init

term1 = Matrix(0.0 * I, 7, 7)
for r in 1:7
      term1[r,:] = (Δ .* Hess[r,:,:]) * p_init
end
term1 + Ω + J_p

test_J = ForwardDiff.jacobian(f_newton, p_init)

f_newton(p_init)


1+1



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 9. Share prediction function
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Write share prediction function
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

function sHat(;δ, X=0.0, σ, ζ=0.0, I_n=1)
      # Fill values if unspecified (default)
      if X == 0.0
            X = repeat(fill(0.0,length(σ))', length(δ))
      end
      if ζ == 0.0
            ζ = reshape(fill(0.0,length(σ)*I_n), I_n, length(σ))
      end

      # Initialize numerator array
      num = Array{Float64}(undef, I_n, length(δ))
      for j in 1:length(δ)
            num_j = BigFloat.(exp.(Ref(δ[j]) .+ ζ * (X[j,:] .* σ)))
            num[:,j] = num_j
      end

      # Construct denominators and shares
      denom = repeat(Ref(BigFloat(1.0)) .+ sum(BigFloat.(num),
                  dims=2), inner=(1,length(δ)))
      ŝ_i = BigFloat.(num) ./ BigFloat.(denom)
      ŝ = 1/(BigFloat(I_n)) * sum(BigFloat.(ŝ_i), dims=1)
      return BigFloat.(ŝ)
end

test_shares = sHat(δ=δ_test, σ=σ_test)
sum(test_shares, dims=2)[1]




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Test cases
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


# First case
J = 3
δ_test = zeros(J);
σ_test = [0.0];
test_shares = sHat(δ=δ_test, σ=σ_test);
println(test_shares)
sum(test_shares, dims=2)[1]

# Second case
J = 3
δ_test = [40, 20, 20];
σ_test = [0.0];
test_shares = sHat(δ=δ_test, σ=σ_test);
println(test_shares)
sum(test_shares, dims=2)[1]


# Third case
J = 3
δ_test = zeros(J);
X_test = repeat([1.0], 3);
σ_test = [0.1];
I_num = 20
ζ_test = reshape(rand(Normal(0,1),length(σ_test)*I_num),
            I_num, length(σ_test))
mean(ζ_test)
test_shares = sHat(δ=δ_test, X=X_test, σ=σ_test, ζ=ζ_test, I_n=I_num);
println(convert.(Float64,round.(test_shares, digits=7)))


X_test = [1.0; 3.0; -1.0];
test_shares = sHat(δ=δ_test, X=X_test, σ=σ_test, ζ=ζ_test, I_n=I_num)
println(convert.(Float64,round.(test_shares, digits=7)))



X_test = [10.0; 10.0; -3.0];
test_shares = sHat(δ=δ_test, X=X_test, σ=σ_test, ζ=ζ_test, I_n=I_num)
println(convert.(Float64,round.(test_shares, digits=7)))









# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 10. Share inversion function
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #















# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 11. Objective function
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #













# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 12. Gradient function
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #














# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 13. Estimation via 2-stage GMM
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
















# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 14. Compare with pyBLP
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #














# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 15. Elasticity matrix for market t=17
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
