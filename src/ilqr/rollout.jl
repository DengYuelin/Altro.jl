
function rollout!(solver::iLQRSolver{T,Q,n}, α) where {T,Q,n}
    Z = solver.Z; Z̄ = solver.Z̄
    Λ = solver.Λ
    K = solver.K; d = solver.d;
    model = solver.model

    Z̄[1].z = [solver.x0; control(Z[1])]

    temp = 0.0
	δx = solver.S[end].q
	δu = solver.S[end].r

    for k = 1:solver.N-1
        δx .= RobotDynamics.state_diff(solver.model, state(Z̄[k]), state(Z[k]))
		δu .= d[k] .* α
		mul!(δu, K[k], δx, 1.0, 1.0)
        ū = control(Z[k]) + δu
        RobotDynamics.set_control!(Z̄[k], ū)

        # Z̄[k].z = [state(Z̄[k]); control(Z[k]) + δu]
        if is_MC_model(model)            
            try
                x⁺, Λ[k] = discrete_dynamics_MC(Q, model, state(Z̄[k]), control(Z̄[k]), Z̄[k].t, Z̄[k].dt)
                Z̄[k+1].z = [x⁺; control(Z[k+1])]
            catch
                println("catch: discrete_dynamics_MC")
                return false
            end
            
            if ~(is_converged(model, state(Z̄[k+1])))
                return false
            end
        else # rc model
            Z̄[k+1].z = [RobotDynamics.discrete_dynamics(Q, solver.model, Z̄[k]);
            control(Z[k+1])]
        end

        max_x = norm(state(Z̄[k+1]),Inf)
        if max_x > solver.opts.max_state_value || isnan(max_x)
            solver.stats.status = STATE_LIMIT
            return false
        end
        max_u = norm(control(Z̄[k+1]),Inf)
        if max_u > solver.opts.max_control_value || isnan(max_u)
            solver.stats.status = CONTROL_LIMIT 
            return false
        end
    end
    solver.stats.status = UNSOLVED
    return true
end

"Simulate the forward the dynamics open-loop"
function rollout!(solver::iLQRSolver{<:Any,Q}) where Q
    rollout!(Q, solver.model, solver.Z, SVector(solver.x0))
    for k in eachindex(solver.Z)
        solver.Z̄[k].t = solver.Z[k].t
    end
end
