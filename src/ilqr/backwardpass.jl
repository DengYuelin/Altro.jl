
"""
Calculates the optimal feedback gains K,d as well as the 2nd Order approximation of the
Cost-to-Go, using a backward Riccati-style recursion. (non-allocating)
"""
function backwardpass!(solver::iLQRSolver{T,QUAD,L,O,n,n̄,m,p,nk,L1,D̄}) where {T,QUAD<:QuadratureRule,L,O,n,n̄,m,p,nk,L1,D̄}
	to = solver.stats.to
    
	N = solver.N

    # Objective
    obj = solver.obj
    model = solver.model

    # Extract variables
    Z = solver.Z; K = solver.K; d = solver.d;
    G = solver.G
    S = solver.S
	Quu_reg = solver.Quu_reg
	Qux_reg = solver.Qux_reg

    # Terminal cost-to-go
	Q = solver.E[N]
    S[N].Q .= Q.Q
    S[N].q .= Q.q

    # Initialize expecte change in cost-to-go
    ΔV = @SVector zeros(2)

	# in the new derivation, these are used in mc model 
    β::Float64 = 1e-6    
	nx::Int = n̄   # notice the LQR is working with error state 
	nu::Int = m
    ncxu::Int = 0
    nm1::Int = 0
    idx2::Int = 0
    idx3::Int = 0
    idx4::Int = 0
    idx5::Int = 0

	if is_MC_model(model)
		ncxu = p
		nm1 = N-1
		idx2 = nu+1
		idx3 = nu+ncxu
		idx4 = nu+ncxu+1
		idx5 = nk
	end

	k = N-1
	while k > 0
		if is_MC_model(model)
			# new derivation, tmp data structures are in iLQRSolver
			cost_exp = solver.E[k]
			dyn_exp = solver.D[k]
			# assume no H
            Q,q,R,r = cost_exp.Q,cost_exp.q,cost_exp.R,cost_exp.r
            A,B,C,dG = dyn_exp.A, dyn_exp.B, dyn_exp.C, dyn_exp.G    # this is already a G in solver (state difference jacobian)
            Vxx = S[k+1].Q
            vx = S[k+1].q

			solver.tmp_nunu[1] .= R
            mul!(solver.tmp_nxnu[1], Vxx, B)  # Vxx*B 
            mul!(solver.tmp_nunu[1], Transpose(B), solver.tmp_nxnu[1], 1.0, 1.0)  # R+B'*Vxx*B
            solver.M[1:nu,1:nu] .= solver.tmp_nunu[1]

            mul!(solver.tmp_nxncxu[1], Vxx, C)  # Vxx*C 
            mul!(solver.tmp_nuncxu[1], Transpose(B), solver.tmp_nxncxu[1])  # B'Vxx*C 
            solver.M[1:nu,idx2:idx3] .= solver.tmp_nuncxu[1]
            solver.M[idx2:idx3,1:nu] .= Transpose(solver.tmp_nuncxu[1])


            mul!(solver.tmp_nuncxu[1], Transpose(B), Transpose(dG))  # B'*G' 
            solver.M[1:nu,idx4:idx5] .= solver.tmp_nuncxu[1]
            solver.M[idx4:idx5,1:nu] .= Transpose(solver.tmp_nuncxu[1])


            mul!(solver.tmp_ncxuncxu[1], Transpose(C), Transpose(dG))  # C'*G' 
            solver.M[idx2:idx3,idx4:idx5] .= solver.tmp_ncxuncxu[1]
            solver.M[idx4:idx5,idx2:idx3] .= Transpose(solver.tmp_ncxuncxu[1])

            solver.tmp_ncxuncxu[1] .= 0
            for j=1:ncxu
                solver.tmp_ncxuncxu[1][j,j] = 1.0
            end
            solver.tmp_ncxuncxu[1] .*= β

        # @time begin
            solver.tmp_ncxuncxu[2] .= solver.tmp_ncxuncxu[1]   # β*I(ncxu)
            mul!(solver.tmp_ncxuncxu[2], Transpose(C), solver.tmp_nxncxu[1], 1.0, 1.0)  # β*I(ncxu) + C'Vxx*C
            solver.M[idx2:idx3,idx2:idx3] .= solver.tmp_ncxuncxu[2]

            solver.tmp_ncxuncxu[1] .*= -1.0
            solver.M[idx4:idx5,idx4:idx5] .= solver.tmp_ncxuncxu[1]   # -β*I(ncxu)


            mul!(solver.tmp_nxnx[1], Vxx, A)  # Vxx*A
            mul!(solver.tmp_nunx[1], Transpose(B), solver.tmp_nxnx[1])  # B'Vxx*A
            mul!(solver.tmp_ncxunx[1], Transpose(C), solver.tmp_nxnx[1])  # C'Vxx*A
            mul!(solver.tmp_ncxunx[2], dG, A)  # G*A
            solver.mb[1:nu,:] .= solver.tmp_nunx[1]
            solver.mb[idx2:idx3,:] .= solver.tmp_ncxunx[1]
            solver.mb[idx4:idx5,:] .= solver.tmp_ncxunx[2]

            solver.K_all .= solver.M\solver.mb
            for ii=1:m
                for jj=1:nx
                    solver.Ku[ii,jj] = solver.K_all[ii,jj]
                end
            end
            
            for ii=1:p
                for jj=1:nx
                    solver.Kλ[ii,jj] = solver.K_all[m+ii,jj]
                end
            end

            solver.tmp_nu[1] .= r
            mul!(solver.tmp_nu[1], Transpose(B), vx, 1.0 ,1.0)
            solver.md[1:nu] .= solver.tmp_nu[1]

            mul!(solver.tmp_ncxu[1], Transpose(C), vx)
            solver.md[idx2:idx3] .= solver.tmp_ncxu[1]
            
            solver.l_all .= solver.M\solver.md

            # solver.lu .= solver.l_all[1:m]
            for ii=1:m
                solver.lu[ii] = solver.l_all[ii]
            end
            # solver.lλ .= solver.l_all[m .+ (1:p)]
            for ii=1:p
                solver.lλ[ii] = solver.l_all[m+ii]
            end

            solver.kl_list[k] .= solver.lu;
            lmul!(-1.0, solver.kl_list[k])
            solver.Kx_list[k] .= solver.Ku;
            lmul!(-1.0, solver.Kx_list[k])

            solver.Kλ_list[k] .= solver.Kλ;
            lmul!(-1.0, solver.Kλ_list[k])
            solver.kλ_list[k] .= solver.lλ;
            lmul!(-1.0, solver.kλ_list[k])

            solver.Abar .= A 
            mul!(solver.Abar, B, solver.Ku, -1.0, 1.0)  # A - BKu
            mul!(solver.Abar, C, solver.Kλ, -1.0, 1.0)

            solver.bbar .= 0
            mul!(solver.bbar, B, solver.lu, -1.0, 1.0)
            mul!(solver.bbar, C, solver.lλ, -1.0, 1.0)
        # end
        # @time begin
            mul!(solver.tmp_nx[1], B, solver.lu)
            t1 = dot(solver.lu,r)  # compare to -solver.lu'*r, no memory allocation
            t1 = -1.0*t1 
            t1 -= dot(vx,solver.tmp_nx[1])


            mul!(solver.tmp_nu[1], R, solver.lu)
            t2 = dot(solver.lu,solver.tmp_nu[1])     # compare to 0.5*solver.lu'*tmp_nu[1], no memory allocation
            t2 = 0.5*t2
            mul!(solver.tmp_nxnu[1], Vxx, B)
            mul!(solver.tmp_nunu[1], Transpose(B), solver.tmp_nxnu[1])

            mul!(solver.tmp_nu[2], solver.tmp_nunu[1], solver.lu)
            t2 += dot(solver.lu, solver.tmp_nu[2])

        # end
            #S[k].Q .= Q + Ku'*R*Ku + Abar'*solver.Vxx_list[i+1]*Abar + β*Kλ'*Kλ
            S[k].Q .= Q
            mul!(solver.tmp_nunx[1], R, solver.Ku)  # R*Ku
            mul!(solver.tmp_nxnx[1], Transpose(solver.Ku), solver.tmp_nunx[1])  # Ku'*R*Ku
            S[k].Q .+= solver.tmp_nxnx[1]

            mul!(solver.tmp_nxnx[2], Vxx, solver.Abar)  # solver.Vxx_list[i+1]*Abar
            mul!(solver.tmp_nxnx[3], Transpose(solver.Abar), solver.tmp_nxnx[2])  # Abar'*solver.Vxx_list[i+1]*Abar
            S[k].Q .+= solver.tmp_nxnx[3]

            mul!(solver.tmp_nxnx[4], Transpose(solver.Kλ), solver.Kλ) #Kλ'*Kλ
            solver.tmp_nxnx[4] .*= β
            S[k].Q .+= solver.tmp_nxnx[4]

            #S[k].q .= q - Ku'*r + Ku'*R*lu + β*Kλ'*lλ + Abar'*solver.Vxx_list[i+1]*bbar + Abar'*solver.vx_list[i+1]
            S[k].q .= q
            mul!(solver.tmp_nx[1], Transpose(solver.Ku), r)  # Ku'*r
            S[k].q .-= solver.tmp_nx[1]     # q - Ku'*r

            mul!(solver.tmp_nu[1], R, solver.lu)  # R*lu
            mul!(solver.tmp_nx[2], Transpose(solver.Ku), solver.tmp_nu[1])  # Ku'*R*lu
            S[k].q .+= solver.tmp_nx[2]     # q - Ku'*r + Ku'*R*lu

            mul!(solver.tmp_nx[3], Transpose(solver.Kλ), solver.lλ)  # Kλ'*lλ
            solver.tmp_nx[3] .*= β
            S[k].q .+= solver.tmp_nx[3]    # q - Ku'*r + Ku'*R*lu + β*Kλ'*lλ

            mul!(solver.tmp_nx[4], Vxx, solver.bbar)
            mul!(solver.tmp_nx[5], Transpose(solver.Abar), solver.tmp_nx[4]) #Abar'*solver.Vxx_list[i+1]*bbar
            S[k].q .+= solver.tmp_nx[5] 

            mul!(solver.tmp_nx[1], Transpose(solver.Abar), vx)
            S[k].q .+= solver.tmp_nx[1]                            # + Abar'*solver.vx_list[i+1]
            
            ΔV += @SVector [t1, t2]

			#output to K and d
			K[k] .= solver.Kx_list[k]
			d[k] .= solver.kl_list[k]

		else
			ix = Z[k]._x
			iu = Z[k]._u

			# Get error state expanions
			fdx,fdu = TO.error_expansion(solver.D[k], model)
			cost_exp = solver.E[k]
			Q = solver.Q_tmp 

			# Calculate action-value expansion
			_calc_Q!(Q, cost_exp, S[k+1], fdx, fdu, S[k])

			# Regularization
			Quu_reg .= Q.R #+ solver.ρ[1]*I
			Quu_reg .+= solver.ρ[1]*Diagonal(@SVector ones(m))
			Qux_reg .= Q.H

			if solver.opts.bp_reg
				vals = eigvals(Hermitian(Quu_reg))
				if minimum(vals) <= 0
					@warn "Backward pass regularized"
					regularization_update!(solver, :increase)
					k = N-1
					ΔV = @SVector zeros(2)
					continue
				end
			end

			# Compute gains
			_calc_gains!(K[k], d[k], Quu_reg, Qux_reg, Q.r)

			# Calculate cost-to-go (using unregularized Quu and Qux)
			ΔV += _calc_ctg!(S[k], Q, K[k], d[k])
		end
	
        k -= 1
    end

    regularization_update!(solver, :decrease)

    return ΔV

end

function static_backwardpass!(solver::iLQRSolver{T,QUAD,L,O,n,n̄,m,p,nk,L1,D̄}, grad_only=false) where {T,QUAD<:QuadratureRule,L,O,n,n̄,m,p,nk,L1,D̄}
	N = solver.N

    # Objective
    obj = solver.obj
    model = solver.model

    # Extract variables
    Z = solver.Z; K = solver.K; d = solver.d;
    G = solver.G
    S = solver.S
	Quu_reg = SMatrix(solver.Quu_reg)
	Qux_reg = SMatrix(solver.Qux_reg)

    # Terminal cost-to-go
	# Q = error_expansion(solver.Q[N], model)
	Q = solver.E[N]
	Sxx = SMatrix(Q.Q)
	Sx = SVector(Q.q)

	if solver.opts.save_S
		S[end].Q .= Sxx
		S[end].q .= Sx
	end

    # Initialize expected change in cost-to-go
	ΔV = @SVector zeros(2)
	
	k = N-1
    while k > 0
        ix = Z[k]._x
        iu = Z[k]._u

		# Get error state expanions
		fdx,fdu = TO.error_expansion(solver.D[k], model)
		fdx,fdu = SMatrix(fdx), SMatrix(fdu)
		Q = TO.static_expansion(solver.E[k])

		# Calculate action-value expansion
		Q = _calc_Q!(Q, Sxx, Sx, fdx, fdu, grad_only)

		# Save Q
		solver.Q[k].Q .= Q.xx
		solver.Q[k].R .= Q.uu
		solver.Q[k].H .= Q.ux
		solver.Q[k].q .= Q.x
		solver.Q[k].r .= Q.u

		# Regularization
		Quu_reg, Qux_reg = _bp_reg!(Q, fdx, fdu, solver.ρ[1], solver.opts.bp_reg_type)

	    if solver.opts.bp_reg
	        vals = eigvals(Hermitian(Quu_reg))
	        if minimum(vals) <= 0
	            @warn "Backward pass regularized"
	            regularization_update!(solver, :increase)
	            k = N-1
	            ΔV = @SVector zeros(2)
	            continue
	        end
	    end

        # Compute gains
		K_, d_ = _calc_gains!(K[k], d[k], Quu_reg, Qux_reg, Q.u, grad_only)

		# Calculate cost-to-go (using unregularized Quu and Qux)
		Sxx, Sx, ΔV_ = _calc_ctg!(Q, K_, d_, grad_only)
		# k >= N-2 && println(diag(Sxx))
		if solver.opts.save_S
			S[k].Q .= Sxx
			S[k].q .= Sx
			S[k].c = ΔV_[1]
		end
		ΔV += ΔV_
        k -= 1
    end

    regularization_update!(solver, :decrease)

    return ΔV
end

function _bp_reg!(Quu_reg::SizedMatrix{m,m}, Qux_reg, Q, fdx, fdu, ρ, ver=:control) where {m}
    if ver == :state
        Quu_reg .= Q.uu #+ solver.ρ[1]*fdu'fdu
		mul!(Quu_reg, Transpose(fdu), fdu, ρ, 1.0)
        Qux_reg .= Q.ux #+ solver.ρ[1]*fdu'fdx
		mul!(Qux_reg, fdu', fdx, ρ, 1.0)
    elseif ver == :control
        Quu_reg .= Q.uu #+ solver.ρ[1]*I
		Quu_reg .+= ρ*Diagonal(@SVector ones(m))
        Qux_reg .= Q.ux
    end
end

function _bp_reg!(Q, fdx, fdu, ρ, ver=:control)
    if ver == :state
		Quu_reg = Q.uu + ρ * fdu'fdu
		Qux_reg = Q.ux + ρ * fdu'fdx
    elseif ver == :control
		Quu_reg = Q.uu + ρ * I
        Qux_reg = Q.ux
    end

	Quu_reg, Qux_reg
end

function _calc_Q!(Q, cost_exp, S1, fdx, fdu, Q_tmp)
	# Compute the cost-to-go, stashing temporary variables in S[k]
	# Qx =  Q.x[k] + fdx'S.x[k+1]
	mul!(Q.q, Transpose(fdx), S1.q)
	Q.q .+= cost_exp.q

    # Qu =  Q.u[k] + fdu'S.x[k+1]
	mul!(Q.r, Transpose(fdu), S1.q)
	Q.r .+= cost_exp.r

    # Qxx = Q.xx[k] + fdx'S.xx[k+1]*fdx
	mul!(Q_tmp.Q, Transpose(fdx), S1.Q)
	mul!(Q.Q, Q_tmp.Q, fdx)
	Q.Q .+= cost_exp.Q

    # Quu = Q.uu[k] + fdu'S.xx[k+1]*fdu
	mul!(Q_tmp.H, Transpose(fdu), S1.Q)
	mul!(Q.R, Q_tmp.H, fdu)
	Q.R .+= cost_exp.R

    # Qux = Q.ux[k] + fdu'S.xx[k+1]*fdx
	mul!(Q_tmp.H, Transpose(fdu), S1.Q)
	mul!(Q.H, Q_tmp.H, fdx)
	Q.H .+= cost_exp.H

	return nothing
end

function _calc_Q!(Q::TO.StaticExpansion, Sxx, Sx, fdx::SMatrix, fdu::SMatrix, grad_only=false)
	Qx = Q.x + fdx'Sx
	Qu = Q.u + fdu'Sx
	if grad_only
		Qxx = Q.xx
		Quu = Q.uu
		Qux = Q.ux
	else
		Qxx = Q.xx + fdx'Sxx*fdx
		Quu = Q.uu + fdu'Sxx*fdu
		Qux = Q.ux + fdu'Sxx*fdx
	end
	TO.StaticExpansion(Qx,Qxx,Qu,Quu,Qux)
end


function _calc_gains!(K::SizedArray, d::SizedArray, Quu::SizedArray, Qux::SizedArray, Qu)
	LAPACK.potrf!('U',Quu.data)
	K .= Qux
	d .= Qu
	LAPACK.potrs!('U', Quu.data, K.data)
	LAPACK.potrs!('U', Quu.data, d.data)
	K .*= -1
	d .*= -1
	# return K,d
end

function _calc_gains!(K, d, Quu::SMatrix, Qux::SMatrix, Qu::SVector, grad_only=false)
	if grad_only
		K_ = SMatrix(K)
	else
		K_ = -Quu\Qux
		K .= K_
	end
	d_ = -Quu\Qu
	d .= d_
	return K_,d_
end

function _calc_ctg!(S, Q, K, d)
	# S.x[k]  =  Qx + K[k]'*Quu*d[k] + K[k]'* Qu + Qux'd[k]
	tmp1 = S.r
	S.q .= Q.q
	mul!(tmp1, Q.R, d)
	mul!(S.q, Transpose(K), tmp1, 1.0, 1.0)
	mul!(S.q, Transpose(K), Q.r, 1.0, 1.0)
	mul!(S.q, Transpose(Q.H), d, 1.0, 1.0)

	# S.xx[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'K[k]
	tmp2 = S.H
	S.Q .= Q.Q
	mul!(tmp2, Q.R, K)
	mul!(S.Q, Transpose(K), tmp2, 1.0, 1.0)
	mul!(S.Q, Transpose(K), Q.H, 1.0, 1.0)
	mul!(S.Q, Transpose(Q.H), K, 1.0, 1.0)
	transpose!(Q.Q, S.Q)
	S.Q .+= Q.Q
	S.Q .*= 0.5

    # calculated change is cost-to-go over entire trajectory
	t1 = dot(d, Q.r)
	mul!(Q.r, Q.R, d)
	t2 = 0.5*dot(d, Q.r)
    return @SVector [t1, t2]
end

function _calc_ctg!(Q::TO.StaticExpansion, K::SMatrix, d::SVector, grad_only::Bool=false)
	Sx = Q.x + K'Q.uu*d + K'Q.u + Q.ux'd
	if grad_only
		Sxx = Q.xx
	else
		Sxx = Q.xx + K'Q.uu*K + K'Q.ux + Q.ux'K
		Sxx = 0.5*(Sxx + Sxx')
	end
	t1 = d'Q.u
	t2 = 0.5*d'Q.uu*d
	return Sxx, Sx, @SVector [t1, t2]
end

