def mesh_CoilSupportBeam(Jstress, folder):
    gravity_options  = Jstress._gravity_options
    material_options = Jstress._material_options
    physics_options  = Jstress._physics_options
    problem_options  = Jstress._problem_options
    coil_support = Jstress._coil_support
    support      = Jstress.fem.support         # the SupportBeams instance
    sdofs        = coil_support.support_dofs      # dict pytree
    curves       = Jstress.fem.base_curves_jax
    # base_dofs = [c.dofs for c in fem.base_curves_jax]
    # curves = [CurveXYZFourierJAX(c.quadpoints, base_dofs[i], c.order)
    #           for i, c in enumerate(fem.base_curves_jax)]
    geom = support._beam_geometry(curves, sdofs)     # note: private, but stable
    x_start = np.asarray(geom['x_start'])            # (N_beams, 3)
    x_end   = np.asarray(geom['x_end'])              # (N_beams, 3)
    gamma3  = np.asarray(support._direction_cosine_matrices(geom, sdofs))  # (N,3,3)
    
    
    # In[7]:
    
    
    # Pulling beam radii
    A_all, Iy, Iz, J = support.cross_section_fn(sdofs)
    A = np.concatenate([np.atleast_1d(np.asarray(a)) for a in A_all])  # (N_beams,)
    radii = np.sqrt(A / np.pi)          # area-equivalent radius; for solid_circle this == r_beam
    
    
    # In[8]:
    
    
    import gmsh
    gmsh.initialize()
    gmsh.model.add("device")
    occ = gmsh.model.occ
    solids = []
    # --- beams: cylinders ---
    for xs, xe, r in zip(x_start, x_end, radii):
        d = xe - xs
        tag = occ.addCylinder(xs[0], xs[1], xs[2], d[0], d[1], d[2], float(r))
        solids.append((3, tag))
    
    
    # ### Step 2: coil solids
    # Each coil is lofted from its true FEM cross-section (the $w_1 \times w_2$ 
    # rectangle carried on the rotation-minimizing frame), sampled as ring wires 
    # along the centerline — the same geometry `coil-fem` sweeps into TET meshes. 
    # A closed loop cannot be lofted in one OCC pass, so each coil is two half 
    # lofts that the boolean fuse glues back together. Each base coil is then 
    # replicated over **every field period and stellarator-symmetry image** (rotate
    # by $2\pi k/n_{fp}$ about $z$, then flip $\mathrm{diag}(1,-1,-1)$ 
    # when reflecting — the same convention as 
    # `coil_fem.geo.symmetries.apply_symmetries_to_gammas`), reconstructing the complete 
    # physical coilset rather than just the pair of images touched by wraparound 
    # beams.
    
    # In[9]:
    
    
    # ============================================================================
    # Step 2: coils as OCC solids (lofted rectangular sweeps + symmetry images)
    # ============================================================================
    # Each coil solid is built by lofting its *true* FEM cross-section — the
    # w1 x w2 rectangle in the RMF frame, exactly as _rect_sweep_points sweeps it:
    #     x = gamma(phi) + (w1/2) u p(phi) + (w2/2) v q(phi)
    # OCC cannot loft a closed loop in a single pass, so each coil is built as
    # two half lofts; the boolean fuse in Step 3 glues them into one volume.
    #
    # Every base coil is then replicated over all nfp field periods and (when
    # stellsym) the stellarator-reflection image, reconstructing the full
    # physical coilset (see the symmetry-image block below).
    
    meshes = Jstress.fem.meshes          # CoilMeshRectangle, one per base coil
    
    def add_coil_solid(mesh, n_slices=96):
        """Loft one coil into two half solids; returns their (dim, tag) list."""
        fc  = mesh.framed_curve
        phi = np.linspace(0.0, 1.0, n_slices, endpoint=False)
        r0  = np.asarray(fc.curve.gamma_eval(phi))       # (K, 3) centerline
        _, p, q = fc.rotated_frame_eval(phi)             # cross-section frame
        p, q = np.asarray(p), np.asarray(q)
    
        # Same corner convention as the FEM sweep; consistent CCW ordering of the
        # ring wires guarantees the loft does not twist between sections.
        corners = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
        wires = []
        for k in range(n_slices):
            pts = [
                occ.addPoint(*(r0[k] + 0.5 * mesh.w1 * u * p[k]
                                     + 0.5 * mesh.w2 * v * q[k]))
                for (u, v) in corners
            ]
            lines = [occ.addLine(pts[a], pts[(a + 1) % 4]) for a in range(4)]
            wires.append(occ.addWire(lines))
    
        # makeRuled=True: straight facets between rings — matches the linear
        # phi-sweep of the FEM mesh and is robust (no B-spline self-intersection).
        half = n_slices // 2
        out  = []
        out += occ.addThruSections(wires[:half + 1], makeSolid=True, makeRuled=False)
        out += occ.addThruSections(wires[half:] + [wires[0]],
                                   makeSolid=True, makeRuled=False)
        occ.remove([(1, w) for w in wires], recursive=True)  # construction wires
        return out
    
    coil_solids = [add_coil_solid(m) for m in meshes]
    
    # ### Step 3: fuse, reflect and mesh
    # All beam cylinders (including the wraparound/bridge groups) are fused together 
    # with the sector's coils into one per-sector unit, which is then rigidly reflected
    # over every field-period rotation and (when stellsym) the stellarator
    # -reflection image. Each master/partner beam pair is designed so its reflected 
    # copies land at distinct physical locations — no de-duplication needed.
    # All beam cylinders, coil half-lofts, and image coils are fused into a single solid 
    # — the beam endpoints sit on the coil centerlines, so every cylinder overlaps 
    # its coils and the union is watertight and conforming. The fused body 
    # is then tet-meshed; mesh size is bounded by the smallest beam radius and the 
    # coil width, and `MeshSizeFromCurvature` keeps the cylinder walls round.
    
    # ============================================================================
    # Step 3: boolean fuse into one solid + tetrahedral meshing
    # ============================================================================
    # The beam endpoints lie on the coil *centerlines*, so every cylinder is
    # embedded in its two coils and the fuse produces a single conforming volume
    # (internal faces — including the half-loft seams — are dissolved).
    
    fused_solid_1fp = solids + [dt for cs in coil_solids for dt in cs]
    fused_1fp, _ = occ.fuse(fused_solid_1fp[:1], fused_solid_1fp[1:])
    occ.synchronize()
    
    # Reflect the whole per-sector unit (coils + all beams, wraparound/bridge
    # groups included) over all nfp field periods and (when stellsym) the
    # stellarator-reflection image. Convention mirrors coil_fem.geo.symmetries
    # (apply_symmetries_to_gammas): rotate phi = 2*pi*k/nfp about z, then flip
    # diag(1, -1, -1) when reflecting; (k=0, flip=False) is fused_1fp itself.
    nfp, stellsym = support.nfp, support.stellsym
    flip_list = (False, True) if stellsym else (False,)
    flip_Q = np.diag([1.0, -1.0, -1.0])
    
    image_solids = []
    for k in range(nfp):
        phi = 2.0 * np.pi * k / nfp
        c, s = np.cos(phi), np.sin(phi)
        rot_Q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for flip in flip_list:
            if k == 0 and not flip:
                continue  # identity: already in fused_1fp
            Q = (flip_Q @ rot_Q) if flip else rot_Q   # rotate, then flip
            affine = list(np.hstack([Q, np.zeros((3, 1))]).ravel())
            copies = occ.copy(fused_1fp)
            occ.affineTransform(copies, affine)
            image_solids += copies
    
    n_images = nfp * len(flip_list) - 1
    print(f'{len(solids)} beam cylinders, '
          f'{sum(len(c) for c in coil_solids)} base-coil half-lofts, '
          f'{len(image_solids)} image half-lofts '
          f'({n_images} field-period/symmetry images x {len(coil_solids)} base coils)')
    
    # fused_1fp is itself a list of (dim, tag) pairs (occ.fuse's output), so it
    # must be *concatenated* with the other lists, not nested inside another list
    # (nesting is what produced the "Invalid data for input vector of pairs"
    # error -- gmsh expects one flat list of (dim, tag) tuples).
    fused_solid_all = fused_1fp + image_solids
    fused_all, _ = occ.fuse(fused_solid_all[:1], fused_solid_all[1:])
    occ.synchronize()
    
    
    vols = gmsh.model.getEntities(3)
    print(f'{len(vols)} volume(s) after fuse')   # >1 means a part is not touching
    gmsh.model.addPhysicalGroup(3, [t for _, t in vols], name='device')
    
    # Mesh sizing: resolve the beam radii and the coil cross-section.
    r_min = float(radii.min())
    gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_scale * 0.5 * mesh_options['w1'])
    gmsh.option.setNumber('Mesh.MeshSizeMin', mesh_scale * 0.15 * r_min)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 12)  # pts per 2*pi
    gmsh.option.setNumber('Mesh.ElementOrder', 2)  # TET10
    
    t0 = time.time()
    gmsh.model.mesh.generate(3)
    print(f'meshed in {time.time() - t0:.1f} s')
    
    gmsh.write(folder + '/full_mesh.msh')  # portable fallback for a separate dolfinx env
    
    
    # ### Step 4: classify nodes + clamp metadata
    # 
    # Inverse-map every gmsh node back into the base-coil $(\varphi, u, v)$ 
    # parametrisation to separate conductor from beam/steel, evaluate clamp
    # Winkler weights, and write a plain `.npz` (for ``beam_dolfinx.py``) plus
    # a VTU for sanity checking.  Lorentz / Biot–Savart body force is **not**
    # computed here — ``beam_dolfinx.py`` evaluates it at volume quadrature
    # points on the imported mesh.
    # 
    # **Known limitation.** `occ.fuse` dissolves the coil/beam interface, so 
    # tets straddle the conductor boundary. If that matters for validation, 
    # switch upstream to `occ.fragment` (keeps the interface mesh-conforming 
    # and tags volumes directly).
    # 
    
    # ============================================================================
    # Step 4A: inverse map (Option B) — classify nodes as conductor vs beam/steel
    # ============================================================================
    # Push every gmsh node back through the symmetry transform and the sweep map
    #     x = gamma(phi) + (w1/2) u p(phi) + (w2/2) v q(phi)
    # to recover (phi, u, v) in the *base* coil frame. Nodes with |u|,|v| <= 1
    # are conductor; the rest are beam/steel.
    
    import interpax
    from scipy.spatial import cKDTree
    
    # Mesh nodes from the live gmsh session (must run before gmsh.finalize()).
    # getNodes() also returns nodes on the leftover cross-section ring wires, which
    # addThruSections keeps alive despite the occ.remove() in Step 2. Those nodes
    # are absent from full_mesh.msh, so keep only the ones a TET10 actually uses.
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    X = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    node_tags = np.asarray(node_tags, dtype=np.int64)
    
    _, _enodes = gmsh.model.mesh.getElementsByType(11)  # 11 = TET10
    _keep = np.isin(node_tags, np.unique(np.asarray(_enodes, dtype=np.int64)))
    X, node_tags = X[_keep], node_tags[_keep]
    N = X.shape[0]
    print(f'gmsh nodes: {N} (dropped {int((~_keep).sum())} not used by any TET10)')
    
    nfp, stellsym = support.nfp, support.stellsym
    n_base = len(Jstress.fem.meshes)
    flip_list = (False, True) if stellsym else (False,)
    flip_Q = np.diag([1.0, -1.0, -1.0])
    
    # Same ordered transform list as coil_fem.geo.symmetries (identity kept).
    # Q maps base -> image; inverse is Q.T since each Q is orthogonal (det ±1).
    Q_list = []
    for k in range(nfp):
        phi_k = 2.0 * np.pi * k / nfp
        c, s = np.cos(phi_k), np.sin(phi_k)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for flip in flip_list:
            Q_list.append(flip_Q @ rot if flip else rot)
    Q_list = np.asarray(Q_list)          # (n_sym, 3, 3)
    n_sym = Q_list.shape[0]
    
    # Dense centreline + RMF frame samples per base coil (uniform grid required
    # for FramedCurveRMFJAX.rotated_frame_eval; scattered phi uses interpax).
    N_PHI = 4096
    phi_s = np.linspace(0.0, 1.0, N_PHI, endpoint=False)
    w1 = float(mesh_options['w1'])
    w2 = float(mesh_options['w2'])
    dist_cut = 0.5 * np.hypot(w1, w2) * 1.05
    # Tolerance for node selection
    uv_tol = 2e-2
    
    # Per-coil samples in BASE frame (gamma, p, q on the dense grid).
    gamma_s = []   # list of (N_PHI, 3)
    p_s = []
    q_s = []
    for mesh in Jstress.fem.meshes:
        fc = mesh.framed_curve
        g = np.asarray(fc.curve.gamma_eval(phi_s))
        _, p, q = fc.rotated_frame_eval(phi_s)
        gamma_s.append(np.asarray(g))
        p_s.append(np.asarray(p))
        q_s.append(np.asarray(q))
    
    # One cKDTree over all image centreline samples.
    sample_pts = []
    sample_lab = []   # (s, i, phi_index)
    for s in range(n_sym):
        Q = Q_list[s]
        for i in range(n_base):
            pts_img = gamma_s[i] @ Q.T          # (N_PHI, 3)
            sample_pts.append(pts_img)
            labs = np.empty((N_PHI, 3), dtype=np.int32)
            labs[:, 0] = s
            labs[:, 1] = i
            labs[:, 2] = np.arange(N_PHI)
            sample_lab.append(labs)
    sample_pts = np.vstack(sample_pts)
    sample_lab = np.vstack(sample_lab)
    tree = cKDTree(sample_pts)
    print(f'KD-tree samples: {sample_pts.shape[0]} '
          f'({n_sym} sectors x {n_base} coils x {N_PHI})')
    
    dist, nn = tree.query(X, k=3)
    # Candidate labels for each of the 3 nearest samples.
    cand_lab = sample_lab[nn]                 # (N, 3, 3) -> (s, i, phi_idx)
    
    owner_sym = np.full(N, -1, dtype=np.int8)
    owner_coil = np.full(N, -1, dtype=np.int8)
    phi_out = np.zeros(N, dtype=np.float32)
    u_out = np.zeros(N, dtype=np.float32)
    v_out = np.zeros(N, dtype=np.float32)
    
    # Process candidates: for each node, try up to 3 nearest (s, i) pairs.
    # Vectorise Newton refinement per (candidate slot, coil) batch for speed.
    for cand in range(3):
        unset = owner_coil < 0
        if not unset.any():
            break
        idxs = np.where(unset)[0]
        s_c = cand_lab[idxs, cand, 0]
        i_c = cand_lab[idxs, cand, 1]
        phi0 = phi_s[cand_lab[idxs, cand, 2]]
        d_c = dist[idxs, cand]
    
        # Reject far candidates early.
        far = d_c > dist_cut
        # Group remaining by coil index for batched gamma_eval / frame interp.
        keep_mask = ~far
        for i in range(n_base):
            m = keep_mask & (i_c == i)
            if not m.any():
                continue
            sel = idxs[m]
            s_sel = s_c[m]
            phi_guess = phi0[m].astype(np.float64)
    
            # Inverse transform into base frame: y = X @ Q  (Q orthogonal).
            Qs = Q_list[s_sel]                              # (n, 3, 3)
            y = np.einsum('ni,nij->nj', X[sel], Qs)         # (n, 3)
    
            # 2 Newton steps on g(phi) = (y - gamma).dot(gammadash) = 0.
            fc = Jstress.fem.meshes[i].framed_curve
            curve = fc.curve
            phi_n = phi_guess.copy()
            for _ in range(2):
                g0 = np.asarray(curve.gamma_eval(phi_n, 0))
                g1 = np.asarray(curve.gamma_eval(phi_n, 1))
                g2 = np.asarray(curve.gamma_eval(phi_n, 2))
                dvec = y - g0
                g = np.sum(dvec * g1, axis=1)
                gp = -np.sum(g1 * g1, axis=1) + np.sum(dvec * g2, axis=1)
                phi_n = phi_n - g / np.where(np.abs(gp) < 1e-30, 1e-30, gp)
            phi_n = np.mod(phi_n, 1.0)
    
            # Frame at refined phi via periodic cubic interp off the dense grid.
            p_i = np.asarray(interpax.interp1d(
                phi_n, phi_s, p_s[i], method='cubic2', period=1.0,
            ))
            q_i = np.asarray(interpax.interp1d(
                phi_n, phi_s, q_s[i], method='cubic2', period=1.0,
            ))
            g0 = np.asarray(curve.gamma_eval(phi_n, 0))
            dvec = y - g0
            u = 2.0 * np.sum(dvec * p_i, axis=1) / w1
            v = 2.0 * np.sum(dvec * q_i, axis=1) / w2
            inside = (np.abs(u) <= 1.0 + uv_tol) & (np.abs(v) <= 1.0 + uv_tol)
    
            take = sel[inside]
            owner_sym[take] = s_sel[inside].astype(np.int8)
            owner_coil[take] = i
            phi_out[take] = phi_n[inside].astype(np.float32)
            u_out[take] = u[inside].astype(np.float32)
            v_out[take] = v[inside].astype(np.float32)
    
    n_cond = int((owner_coil >= 0).sum())
    print(f'conductor nodes: {n_cond} / {N} ({100 * n_cond / N:.1f}%)')
    print(f'beam/steel nodes: {N - n_cond}')
    
    
    # In[15]:
    
    
    # ============================================================================
    # Step 4B: Winkler clamp weights + physical clamp centres
    # ============================================================================
    # Clamp weights are evaluated in the BASE frame (same as Support.compute_weights).
    # Lorentz / Biot–Savart live in beam_dolfinx.py at volume quadrature points.
    
    import jax.numpy as jnp
    from coil_fem.coupling import Support
    
    curves_jax = Jstress.fem.base_curves_jax
    sdofs = coil_support.support_dofs
    support_weight = np.zeros(N, dtype=np.float64)
    
    # Base-frame preimages for conductor nodes (for compute_weights).
    y_base = np.zeros((N, 3), dtype=np.float64)
    cond = owner_coil >= 0
    if cond.any():
        Qs = Q_list[owner_sym[cond]]
        y_base[cond] = np.einsum('ni,nij->nj', X[cond], Qs)
    
    t0 = time.time()
    for i in range(n_base):
        sel = np.where(owner_coil == i)[0]
        if sel.size == 0:
            continue
        y_i = y_base[sel]
        # Clamp-only Winkler weights: call the base Support path with clamp
        # dofs so beam attachment weights stay zero on the fused mesh.
        w_g, _w_a = Support.compute_weights(
            self=support,
            coil_idx=i,
            surface_pts=jnp.asarray(y_i),
            curves_jax=curves_jax,
            dofs={'phis': sdofs['phis']},
        )
        support_weight[sel] = np.asarray(w_g)
        print(f'  coil {i}: {sel.size} nodes')

    # Physical-frame clamp centres for dolfinx (analytic Winkler at facet quads).
    phis_clamp = sdofs['phis']
    r_clamp = float(coil_support._r_clamp)
    eps_sigmoid = float(coil_support._sig_eps)
    clamp_centers = []
    for i in range(n_base):
        phi_i = np.asarray(phis_clamp[i], dtype=np.float64).ravel()
        c_base = np.asarray(
            Jstress.fem.base_curves_jax[i].gamma_eval(phi_i), dtype=np.float64,
        )
        for s in range(n_sym):
            clamp_centers.append(c_base @ Q_list[s].T)
    clamp_centers = np.vstack(clamp_centers)
    print(
        f'clamp_centers: {clamp_centers.shape[0]} '
        f'(n_base={n_base} x n_sym={n_sym} x n_clamp='
        f'{clamp_centers.shape[0] // (n_base * n_sym)}), '
        f'r_clamp={r_clamp:.4g}, eps_sigmoid={eps_sigmoid:.4g}'
    )
    print(f'clamp weights assembled in {time.time() - t0:.1f} s')
    print(f'support_weight range: '
          f'[{support_weight.min():.3g}, {support_weight.max():.3g}]')
    
    
    # In[16]:
    
    
    # ============================================================================
    # Step 4C: round-trip check + body_force.npz
    # ============================================================================
    
    # Round-trip: rebuild x from (phi, u, v) for 1000 random conductor nodes
    # and compare to the stored gmsh coordinate (loft chord sagitta ~ mm).
    rng = np.random.default_rng(0)
    cond_idx = np.where(owner_coil >= 0)[0]
    n_check = min(1000, cond_idx.size)
    pick = rng.choice(cond_idx, size=n_check, replace=False)
    
    errs = np.empty(n_check)
    for j, n in enumerate(pick):
        i = int(owner_coil[n])
        s = int(owner_sym[n])
        mesh = Jstress.fem.meshes[i]
        fc = mesh.framed_curve
        phi_n = float(phi_out[n])
        g = np.asarray(fc.curve.gamma_eval(np.array([phi_n])))[0]
        p = np.asarray(interpax.interp1d(
            np.array([phi_n]), phi_s, p_s[i], method='cubic2', period=1.0,
        ))[0]
        q = np.asarray(interpax.interp1d(
            np.array([phi_n]), phi_s, q_s[i], method='cubic2', period=1.0,
        ))[0]
        y = g + 0.5 * w1 * float(u_out[n]) * p + 0.5 * w2 * float(v_out[n]) * q
        x_rec = Q_list[s] @ y
        errs[j] = np.linalg.norm(x_rec - X[n])
    
    print(f'round-trip max |x_rec - X| over {n_check} nodes: {errs.max()*1e3:.3f} mm')
    assert errs.max() < 2e-3, (
        f'round-trip error {errs.max()*1e3:.3f} mm exceeds 2 mm loft-chord budget'
    )
    print(f'conductor fraction: {100 * cond_idx.size / N:.1f}%')
    
    g_vec_save = np.asarray(
        gravity_options.get('g_vec', (0.0, 0.0, 0.0)), dtype=np.float64,
    )
    np.savez(
        folder + '/body_force.npz',
        points=X,
        node_tags=node_tags,
        support_weight=support_weight,
        owner_coil=owner_coil,
        owner_sym=owner_sym,
        phi=phi_out,
        u=u_out,
        v=v_out,
        rho=np.float64(material_options['density']),
        g_vec=g_vec_save,
        k_clamp=np.float64(support.k_clamp),
        E=np.float64(material_options['E']),
        nu=np.float64(material_options['nu']),
        clamp_centers=clamp_centers,
        r_clamp=np.float64(r_clamp),
        eps_sigmoid=np.float64(eps_sigmoid),
    )
    print('wrote body_force.npz (no f_vol / B_*; force computed in beam_dolfinx)')
    
    
    # In[17]:
    
    
    # ============================================================================
    # Step 4D: VTU sanity dump (classification + clamp weights)
    # ============================================================================
    # Connectivity from the live gmsh session. gmsh tet10 edge order
    # (01)(12)(02)(03)(23)(13) -> VTK (01)(12)(02)(03)(13)(23).
    
    import meshio
    
    etags, enodes = gmsh.model.mesh.getElementsByType(11)  # 11 = TET10
    
    inv = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    inv[node_tags] = np.arange(node_tags.size)
    cells = inv[np.asarray(enodes, dtype=np.int64).reshape(-1, 10)][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    
    meshio.Mesh(
        points=X,
        cells=[('tetra10', cells)],
        point_data={
            'w_clamp': support_weight,
            'k_clamp_Npm3': support_weight * float(support.k_clamp),
            'owner_coil': owner_coil.astype(np.int32),   # -1 = beam/steel
            'owner_sym': owner_sym.astype(np.int32),
        },
    ).write(folder + '/full_body_fields.vtu')
    print(f'wrote full_body_fields.vtu ({cells.shape[0]} tet10 cells, {N} nodes)')
    
    
    # ### Step 5: import into dolfinx
    # `dolfinx.io.gmshio.model_to_mesh` converts the live gmsh model directly 
    # (the physical group added in Step 3 is required — dolfinx only imports 
    # tagged cells). If dolfinx is installed in a different environment than 
    # gmsh/coil-fem, use `gmshio.read_from_msh('device.msh', ...)` there instead.
    # 
    
    # In[18]:
    
    
    # ============================================================================
    # Step 5: import into dolfinx
    # ============================================================================
    # Reads the mesh straight from the live gmsh model (no file round-trip).
    # If dolfinx lives in a different environment, run there instead:
    #     domain, ct, ft = gmshio.read_from_msh('device.msh', MPI.COMM_WORLD,
    #                                           rank=0, gdim=3)
    from mpi4py import MPI
    from dolfinx.io import gmshio
    
    domain, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3,
    )
    gmsh.finalize()
    
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)
    print('cells:', domain.topology.index_map(tdim).size_global)
    print('nodes:', domain.geometry.x.shape[0])
    
