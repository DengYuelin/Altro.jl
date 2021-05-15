
"""
    iLQRSolver

A fast solver for unconstrained trajectory optimization that uses a Riccati recursion
to solve for a local feedback controller around the current trajectory, and then 
simulates the system forward using the derived feedback control law.

# Constructor
    Altro.iLQRSolver(prob, opts; kwarg_opts...)
"""
struct iLQRSolver{T,I<:QuadratureRule,L,O,n,n̄,m,p,nk,L1,D̄} <: UnconstrainedSolver{T}
    # Model + Objective
    model::L
    obj::O

    # Problem info
    x0::MVector{n,T}
    xf::MVector{n,T}
    tf::T
    N::Int

    opts::SolverOptions{T}
    stats::SolverStats{T}

    # Primal Duals
    Z::Traj{n,m,T,KnotPoint{T,n,m,L1}}
    Z̄::Traj{n,m,T,KnotPoint{T,n,m,L1}}

    # For max coords models
    Λ::Vector{Vector{T}}
    Kx_list::Vector{SizedMatrix{m,n̄,T,2,Matrix{T}}}   # 1 --- N-1   
    kl_list::Vector{SizedVector{m,T,Vector{T}}}       # 1 --- N-1
    Kλ_list::Vector{SizedMatrix{p,n̄,T,2,Matrix{T}}}   # 1 --- N-1   
    kλ_list::Vector{SizedVector{p,T,Vector{T}}}       # 1 --- N-1
    M::SizedMatrix{nk,nk,T,2,Matrix{T}}
    mb::SizedMatrix{nk,n̄,T,2,Matrix{T}}
    md::SizedVector{nk,T,Vector{T}} 
    K_all::SizedMatrix{nk,n̄,T,2,Matrix{T}}
    l_all::SizedVector{nk,T,Vector{T}} 
    Ku::SizedMatrix{m,n̄,T,2,Matrix{T}}
    Kλ::SizedMatrix{p,n̄,T,2,Matrix{T}}
    lu::SizedVector{m,T,Vector{T}} 
    lλ::SizedVector{p,T,Vector{T}} 
    Abar::SizedMatrix{n̄,n̄,T,2,Matrix{T}}
    bbar::SizedVector{n̄,T,Vector{T}}
    
    tmp_nxnx::Vector{SizedMatrix{n̄,n̄,T,2,Matrix{T}}}
    tmp_nunu::Vector{SizedMatrix{m,m,T,2,Matrix{T}}}
    tmp_nxnu::Vector{SizedMatrix{n̄,m,T,2,Matrix{T}}}
    tmp_nunx::Vector{SizedMatrix{m,n̄,T,2,Matrix{T}}}
    tmp_nxncxu::Vector{SizedMatrix{n̄,p,T,2,Matrix{T}}}
    tmp_ncxunx::Vector{SizedMatrix{p,n̄,T,2,Matrix{T}}}
    tmp_nuncxu::Vector{SizedMatrix{m,p,T,2,Matrix{T}}}
    tmp_ncxunu::Vector{SizedMatrix{p,m,T,2,Matrix{T}}}
    tmp_ncxuncxu::Vector{SizedMatrix{p,p,T,2,Matrix{T}}}

    tmp_nx::Vector{SizedVector{n̄,T,Vector{T}} }
    tmp_nu::Vector{SizedVector{m,T,Vector{T}} }
    tmp_ncxu::Vector{SizedVector{p,T,Vector{T}} }
    

    # Data variables
    # K::Vector{SMatrix{m,n̄,T,L2}}  # State feedback gains (m,n,N-1)
    K::Vector{SizedMatrix{m,n̄,T,2,Matrix{T}}}  # State feedback gains (m,n,N-1)
    d::Vector{SizedVector{m,T,Vector{T}}}  # Feedforward gains (m,N-1)

    D::Vector{D̄}
    G::Vector{SizedMatrix{n,n̄,T,2,Matrix{T}}}        # state difference jacobian (n̄, n)

	quad_obj::QuadraticObjective{n,m,T}  # quadratic expansion of obj
	S::QuadraticObjective{n̄,m,T}         # Cost-to-go expansion
    E::QuadraticObjective{n̄,m,T}         # cost expansion 
    Q::QuadraticObjective{n̄,m,T}         # Action-value expansion
    Qprev::QuadraticObjective{n̄,m,T}     # Action-value expansion from previous iteration

    Q_tmp::TO.QuadraticCost{n̄,m,T,SizedMatrix{n̄,n̄,T,2,Matrix{T}},SizedMatrix{m,m,T,2,Matrix{T}}}
	Quu_reg::SizedMatrix{m,m,T,2,Matrix{T}}
	Qux_reg::SizedMatrix{m,n̄,T,2,Matrix{T}}
    ρ::Vector{T}   # Regularization
    dρ::Vector{T}  # Regularization rate of change

    cache::FiniteDiff.JacobianCache{Vector{T}, Vector{T}, Vector{T}, UnitRange{Int}, Nothing, Val{:forward}(), T}
    grad::Vector{T}  # Gradient

    logger::SolverLogger

end

function iLQRSolver(
        prob::Problem{QUAD,T}, 
        opts::SolverOptions=SolverOptions(), 
        stats::SolverStats=SolverStats(parent=solvername(iLQRSolver));
        kwarg_opts...
    ) where {QUAD,T}
    set_options!(opts; kwarg_opts...)

    # Init solver results
    n,m,N = size(prob)
    n̄ = RobotDynamics.state_diff_size(prob.model)
    p = 0

    x0 = prob.x0
    xf = prob.xf

    Z = prob.Z
    # Z̄ = Traj(n,m,Z[1].dt,N)
    Z̄ = copy(prob.Z)

	K = [zeros(T,m,n̄) for k = 1:N-1]
    d = [zeros(T,m)   for k = 1:N-1]

    D = [DynamicsExpansion{T}(n,n̄,m) for k = 1:N-1]
	G = [SizedMatrix{n,n̄}(zeros(n,n̄)) for k = 1:N+1]  # add one to the end to use as an intermediate result
    
    # for mc
    Λ = [[]]

    nx = n̄
    nu = m
    ncxu = 2 # for non-mc models, use a default magic number?
    if is_MC_model(prob.model)
        ncxu = prob.model.p
    end
    nk = nu+2*ncxu
    Kx_list = [SizedMatrix{nu,nx}(zeros(T,nu,nx)) for i=1:N]
    kl_list = [SizedVector{nu}(zeros(T,nu)) for i=1:N]
    Kλ_list = [SizedMatrix{ncxu,nx}(zeros(T,ncxu,nx)) for i=1:N]
    kλ_list = [SizedVector{ncxu}(zeros(T,ncxu)) for i=1:N]
    M = SizedMatrix{nk,nk}(zeros(T,nk,nk))
    mb = SizedMatrix{nk,nx}(zeros(T,nk,nx))
    md = SizedVector{nk}(zeros(T,nk))
    K_all = SizedMatrix{nk,nx}(zeros(T,nk,nx))
    l_all = SizedVector{nk}(zeros(T,nk))
    Ku = SizedMatrix{nu,nx}(zeros(T,nu,nx))
    Kλ = SizedMatrix{ncxu,nx}(zeros(T,ncxu,nx))
    lu = SizedVector{nu}(zeros(T,nu))
    lλ = SizedVector{ncxu}(zeros(T,ncxu))
    Abar = SizedMatrix{nx,nx}(zeros(T,nx,nx))
    bbar = SizedVector{nx}(zeros(T,nx))

    tmp_nxnx = [SizedMatrix{nx,nx}(zeros(T,nx,nx)) for i=1:5]
    tmp_nunu = [SizedMatrix{nu,nu}(zeros(T,nu,nu)) for i=1:5]
    tmp_nxnu = [SizedMatrix{nx,nu}(zeros(T,nx,nu)) for i=1:5]
    tmp_nunx = [SizedMatrix{nu,nx}(zeros(T,nu,nx)) for i=1:5]
    tmp_nxncxu = [SizedMatrix{nx,ncxu}(zeros(T,nx,ncxu)) for i=1:5]
    tmp_ncxunx = [SizedMatrix{ncxu,nx}(zeros(T,ncxu,nx)) for i=1:5]
    tmp_nuncxu = [SizedMatrix{nu,ncxu}(zeros(T,nu,ncxu)) for i=1:5]
    tmp_ncxunu = [SizedMatrix{ncxu,nu}(zeros(T,ncxu,nu)) for i=1:5]
    tmp_ncxuncxu = [SizedMatrix{ncxu,ncxu}(zeros(T,ncxu,ncxu)) for i=1:5]
    tmp_nx = [SizedVector{nx}(zeros(T,nx))  for i=1:5]
    tmp_nu = [SizedVector{nu}(zeros(T,nu))  for i=1:5]
    tmp_ncxu = [SizedVector{ncxu}(zeros(T,ncxu))  for i=1:5]   

    if is_MC_model(prob.model)
        D = [TO.DynamicsExpansionMC(prob.model) for k = 1:N-1]
        p = prob.model.p # add function for constraint size?
        Λ = [zeros(p) for k = 1:N]
    end
    
    E = QuadraticObjective(n̄,m,N)
	quad_exp = QuadraticObjective(E, prob.model)
	Q = QuadraticObjective(n̄,m,N)
	Qprev = QuadraticObjective(n̄,m,N)
	S = QuadraticObjective(n̄,m,N)

    Q_tmp = TO.QuadraticCost{T}(n̄,m)
	Quu_reg = SizedMatrix{m,m}(zeros(m,m))
	Qux_reg = SizedMatrix{m,n̄}(zeros(m,n̄))
    ρ = zeros(T,1)
    dρ = zeros(T,1)

    cache = FiniteDiff.JacobianCache(prob.model)
    grad = zeros(T,N-1)

    logger = SolverLogging.default_logger(opts.verbose >= 2)
	L = typeof(prob.model)
	O = typeof(prob.obj)
    solver = iLQRSolver{T,QUAD,L,O,n,n̄,m,p,nk,n+m,eltype(D)}(prob.model, prob.obj, x0, xf,
		prob.tf, N, opts, stats,
        Z, Z̄, Λ, 
        Kx_list, kl_list, Kλ_list, kλ_list,
        M,mb,md,
        K_all, l_all,
        Ku, Kλ, lu, lλ,
        Abar, bbar,
        tmp_nxnx, tmp_nunu,
        tmp_nxnu, tmp_nunx,
        tmp_nxncxu, tmp_ncxunx, tmp_nuncxu, tmp_ncxunu,
        tmp_ncxuncxu,
        tmp_nx, tmp_nu, tmp_ncxu,
        K, d, D, G, quad_exp, S, E, Q, Qprev, Q_tmp, Quu_reg, Qux_reg, ρ, dρ, 
        cache, grad, logger)

    reset!(solver)
    return solver
end

# Getters
Base.size(solver::iLQRSolver{<:Any,<:Any,<:Any,<:Any,n,<:Any,m,<:Any,<:Any,<:Any,<:Any}) where {n,m} = n,m,solver.N
@inline TO.get_trajectory(solver::iLQRSolver) = solver.Z
@inline TO.get_objective(solver::iLQRSolver) = solver.obj
@inline TO.get_model(solver::iLQRSolver) = solver.model
@inline get_initial_state(solver::iLQRSolver) = solver.x0
@inline TO.integration(solver::iLQRSolver{<:Any,Q}) where Q = Q
solvername(::Type{<:iLQRSolver}) = :iLQR

log_level(::iLQRSolver) = InnerLoop

function reset!(solver::iLQRSolver{T}) where T
    reset_solver!(solver)
    solver.ρ[1] = 0.0
    solver.dρ[1] = 0.0
    return nothing
end

