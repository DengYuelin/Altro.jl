
# @inline TO.DynamicsExpansionMC(model::AbstractModelMC) = TO.DynamicsExpansionMC{Float64}(model)
# @inline function TO.DynamicsExpansionMC{T}(model::AbstractModelMC) where T
# 	n,m = size(model)
# 	n̄ = state_diff_size(model)
# 	TO.DynamicsExpansionMC{T}(n,n̄,m,model.p)
# end
# function TO.dynamics_expansion!(Q, D::Vector{<:TO.DynamicsExpansionMC}, model::AbstractModelMC,
#   Z::Traj)
#   for k in eachindex(D)
#     D[k].A,D[k].B,D[k].C,D[k].G = discrete_jacobian_MC(Q, model, Z[k])

#   end
# end
# @inline TO.error_expansion(D::TO.DynamicsExpansionMC, model::AbstractModelMC) = D.A, D.B, D.C, D.G

# function TO.error_expansion!(D::Vector{<:TO.DynamicsExpansionMC}, model::AbstractModelMC, G)
# 	return
# end

@inline is_converged(::AbstractModel,x) = throw(ErrorException("is_converged not implemented"))
@inline discrete_jacobian_MC(Q, ::AbstractModel, z) = throw(ErrorException("discrete_jacobian_MC not implemented"))

@inline TO.DynamicsExpansionMC(model) = TO.DynamicsExpansionMC{Float64}(model)
@inline function TO.DynamicsExpansionMC{T}(model) where T
	n,m = size(model)
	n̄ = state_diff_size(model)
	TO.DynamicsExpansionMC{T}(n,n̄,m,model.p)
end
function TO.dynamics_expansion!(Q, D::Vector{<:TO.DynamicsExpansionMC}, model,
  Z::Traj)
  for k in eachindex(D)
    D[k].A,D[k].B,D[k].C,D[k].G = discrete_jacobian_MC(Q, model, Z[k])

  end
end
@inline TO.error_expansion(D::TO.DynamicsExpansionMC, model) = D.A, D.B, D.C, D.G

function TO.error_expansion!(D::Vector{<:TO.DynamicsExpansionMC}, model, G)
	return
end