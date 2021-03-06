## MAX COORD SPECIFIC FUNCTIONS
@inline is_MC_model(model::AbstractModel) = (model isa AbstractModelMC) || (model isa RigidBodyMC) || (model isa LieGroupModelMC) 

@inline is_converged(::AbstractModel,x) = throw(ErrorException("is_converged not implemented"))

@inline discrete_dynamics_MC(Q, model,  x, u, t, dt)= throw(ErrorException("discrete_dynamics_MC not implemented"))

# @inline discrete_jacobian_MC!(Q, Dexp, model, z) = throw(ErrorException("discrete_jacobian_MC not implemented"))
@inline discrete_jacobian_MC!(Q, Dexp, model, z, x⁺, λ) = throw(ErrorException("discrete_jacobian_MC not implemented"))

@inline config_size(model) = throw(ErrorException("not implemented"))

function mc_dims(model)
	nq = config_size(model)
	nv = RD.state_dim(model) - nq
	nc = model.p
	return nq, nv, nc
end

## MIRRORED FUNCTIONS FROM TO - expansions.jl
function TO.save_tmp!(D::TO.DynamicsExpansionMC)
	D.tmpA .= D.A_
	D.tmpB .= D.B_
  	D.tmpC .= D.C_
end

# call RC version if model is RC
TO.dynamics_expansion!(Q, D::Vector{<:TO.DynamicsExpansion}, model, Z::Traj, Λ) =
	TO.dynamics_expansion!(Q, D, model, Z) 

function TO.dynamics_expansion!(Q, D::Vector{<:TO.DynamicsExpansionMC}, model, 
	Z::Traj, Λ)
    for k in eachindex(D)
		if Z[k].dt == 0
			z = copy(Z[k])
			z.dt = Z[1].dt
			discrete_jacobian_MC!(Q, D[k], model, z, state(Z[k+1]), Λ[k])
		else
			discrete_jacobian_MC!(Q, D[k], model, Z[k], state(Z[k+1]), Λ[k])
		end
    end
end

function TO.error_expansion!(D::TO.DynamicsExpansionMC, G1, G2)
    mul!(D.tmp, D.tmpA, G1)
    mul!(D.A, Transpose(G2), D.tmp)
    mul!(D.B, Transpose(G2), D.tmpB)
    mul!(D.C, Transpose(G2), D.tmpC)
	return
end

@inline TO.error_expansion(D::TO.DynamicsExpansionMC, model::LieGroupModelMC) = D.A, D.B, D.C, D.G
@inline TO.error_expansion(D::TO.DynamicsExpansionMC, model::RigidBodyMC) = D.A, D.B, D.C, D.G
@inline TO.error_expansion(D::TO.DynamicsExpansionMC, model) = D.tmpA, D.tmpB, D.tmpC, D.G

@inline TO.DynamicsExpansionMC(model) = TO.DynamicsExpansionMC{Float64}(model)
@inline function TO.DynamicsExpansionMC{T}(model) where T
	n,m = size(model)
	n̄ = RobotDynamics.state_diff_size(model)
	TO.DynamicsExpansionMC{T}(n,n̄,m,model.p)
end

function TO.error_expansion!(D::Vector{<:TO.DynamicsExpansionMC}, model::AbstractModel, G)
	for d in D
		save_tmp!(d)
	end
end

function TO.error_expansion!(D::Vector{<:TO.DynamicsExpansionMC}, model::LieGroupModelMC, G)
	for k in eachindex(D)
		TO.save_tmp!(D[k])
		TO.error_expansion!(D[k], G[k], G[k+1])
	end
end

function TO.error_expansion!(D::Vector{<:TO.DynamicsExpansionMC}, model::RigidBodyMC, G)
	for k in eachindex(D)
		TO.save_tmp!(D[k])
		TO.error_expansion!(D[k], G[k], G[k+1])
	end
end