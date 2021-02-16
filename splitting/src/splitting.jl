module splitting

using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation
using IntervalTrees
using SparseArrays
using NearestNeighbors
using DataStructures
using OrderedCollections
using LinearAlgebra
using Base.Threads


function spaceindex(model::Lar.LAR)::Array{Array{Int,1},1}
	V,CV = model[1:2]
	# se il modello è in 3d o 2d (guardo le righe di V, in 3d V è una 3xN, in 2d V è una 2xN)
	dim = size(V,1)
	#PARALLELIZZO LA CREAZIONE DEI CELLPOINTS
	n=length(CV)
	cellpoints = Array{Array{Float64,2}}(undef,n)
	@inbounds @threads for k=1:n
		cellpoints[k] = V[:,CV[k]]::Lar.Points
	end
	#PARALLELIZZO LA CREAZIONE DEI BOUNDING BOXES
	bboxes = Array{Array{Float64,2}}(undef,n)
	@inbounds @threads for k=1:n
		bboxes[k] = hcat(boundingbox(cellpoints[k])...)
	end
	coverXYZ= Array{Array{Array{Int64,1},1}}(undef,dim)
	#Per ogni asse x=1, y=2, z=3.....
	@threads for i=1:dim
		boxdict = coordintervals(i,bboxes)
		#Creo interval tree sull'asse i
		intTree = IntervalTrees.IntervalMap{Float64, Array}()
		@inbounds for (key,boxset) in boxdict
			intTree[tuple(key...)] = boxset
		end
		coverXYZ[i] = boxcovering(bboxes, i, intTree)     
	end
	spaceindex = Array{Array{Any,1}}(undef,length(bboxes))
	@inbounds @threads for i=1:n
		spaceindex[i] = intersect((coverXYZ[1][i],coverXYZ[2][i])...)
	end
	if(dim==3)
		@inbounds @threads for i=1:n
			spaceindex[i] = intersect((spaceindex[i],coverXYZ[3][i])...)
		end
	end
	@inbounds @simd for k=1:length(spaceindex)
		spaceindex[k] = setdiff(spaceindex[k],[k])
	end
	return spaceindex
end


#Questa funzione prende in input insieme di vertici e calcola il loro bounding box.
#Fa cioè il minimo e il massimo delle loro coordinate su tutti gli assi.
function boundingbox(vertices::Lar.Points)
	d=size(vertices)[1]
	numPoints=size(vertices)[2]
	#inizializzo gli array da ritornare [xMin, yMin, zMin] e [xMax, yMax, zMax]
	mins = zeros(d,1)
	maxs = zeros(d,1)
	for i=1:d
		mins[i]=vertices[i]
		maxs[i]=vertices[i]
	end
	@threads for i=2:numPoints
		@threads for j=1:d
			if(vertices[j+d*(i-1)] > maxs[j])
				maxs[j] = vertices[j+d*(i-1)]
			end
			if(vertices[j+d*(i-1)] < mins[j])
				mins[j] = vertices[j+d*(i-1)]
			end
		end
	end
	
	return (mins,maxs)
end

#Questa funzione computa un dizionario avente come chiave una coordinata [cmin,cmax], e
#come valore un array contenente gli indici di tutte le celle incidenti sulla chiave. 
function coordintervals(coord,bboxes)
	boxdict = OrderedDict{Array{Float64,1},Array{Int64,1}}()
	#Per ogni bounding box...
	l = length(bboxes)
	for h=1:l
		#La chiave del dizionario è [cmin,cmax]
		key = bboxes[h][coord,:]
		#Se il dizionario non ha la chiave, creo la entry..
		if !haskey(boxdict,key)
			boxdict[key] = [h]
		else #Altrimenti pusho la cella, in quanto condividerà [cmin,cmax] con altre celle
			push!(boxdict[key], h)
		end
	end
	return boxdict
end

#Questa funzione computa le intersezioni di celle facendo una ricerca efficiente sugli 
#intervalTrees. index=1 per assex, index=2 per assey, index=3 per assez
function boxcovering(bboxes, index, tree)
	#Inizializzo array vuoti per tutti i box
	n = length(bboxes)
	covers = Array{Array{Int64,1}}(undef, n)
	@inbounds @threads for k=1:n
		covers[k] = []
	end
	#Per ogni bbox....
	@inbounds for (i,boundingbox) in enumerate(bboxes)
		extent = bboxes[i][index,:]
		#Faccio una query all'interval tree su un intervallo 
		iterator = IntervalTrees.intersect(tree, tuple(extent...))
		#Tutte le celle che trovo le appendo all'array del box
		for x in iterator
			append!(covers[i],x.value)
		end
	end
	return covers
end

# Questa funzione fa l'intersezione tra 2 segmenti e ritorna i parametri d'intersezione.
function intersection(line1,line2)
	x1,y1,x2,y2 = vcat(line1...)
	x3,y3,x4,y4 = vcat(line2...)
	det = (x4-x3)*(y1-y2)-(x1-x2)*(y4-y3)
	if det != 0.0
		a = 1/det
		b = [y1-y2 x2-x1; y3-y4 x4-x3]  # x1-x2 => x2-x1 bug in the source link !!
		c = [x1-x3; y1-y3]
		(β,α) = a * b * c
	else
		return ()
	end
	return α,β
end


function linefragments(V,EV,sigma)
	# Inizializzo dati
	m = length(sigma) #Numero segmenti 
	sigma = map(sort,sigma)  #ordino i segmenti intersecati di ogni segmento ([[2,3,....],[1,3,.....])
	params = Array{Array{Float64,1}}(undef,m)
	@threads for i=1:m
		params[i] = []
	end
	line1=[0.0 0.0; 0.0 0.0]
	line2=[0.0 0.0; 0.0 0.0]
	#Per ogni segmento...
	@threads for h=1:m
		#Se il segmento si interseca con qualcosa:
		if sigma[h] ≠ []
			#Prendo i punti(x1,y1),(x2,y2) del segmento h-esimo:
			line1 = V[:,EV[h]]
			#				Confronto il segmento h-esimo con tutti gl altri segmenti
			#presenti nel suo indice spaziale.
			@threads for k in sigma[h]
				#Prendo i punti(x3,y3),(x4,y4) del segmento k-esimo presente in
				line2 = V[:,EV[k]]
				#Ritorno (se esistono) i parametri (alfa,beta) necessari a
				#il punto di intersezione tra coppie di segmenti, h <-> k
				out = intersection(line1,line2) 
				#Se ho intersezione tra le rette (ovvero out non è nothing)
				if out ≠ ()
					#Controllo che i parametri α,β siano ammissibili, se lo sono
					#li immagazzino nella struttura dati params
					if 0<=out[1]<=1 && 0<=out[2]<=1
						push!(params[h], out[1])
						push!(params[k], out[2])
					end
				end
			end
		end
	end
	# Inizializzo struttura da ritornare
	len = length(params)
	@threads for i=1:len
		push!(params[i], 0.0, 1.0) # Aggiungo parametri
		params[i] = sort(collect(Set(params[i]))) #Tolgo i doppioni
	end
	return params
end

function fragmentlines(model)
	V,EV = model
	# Creo indice spaziale
	Sigma = spaceindex(model)
	# calcolo parametri d'intersezione degli spigoli
	lineparams = linefragments(V,EV,Sigma)
	vertdict = OrderedDict{Array{Float64,1},Array{Int,1}}()
	pairs = collect(zip(lineparams, [V[:,e] for e in EV]))
	vertdict = OrderedDict{Array{Float64,1},Int}()
	#Inizializzo nuovi V, EV per aggiungere i nuovi vertici/spigoli dello splitting
	W = Array[]
	EW = Array[]
	k = 0
	l = length(pairs)
	# Ricostruisco i nuovi punti generati dall'intersezione tra spigoli
	# tramite i parametri d'intersezione
	# Per ogni spigolo...
	@inbounds @simd for i = 1:l
		params = pairs[i][1]
		linepoints = pairs[i][2]
		v1 = linepoints[:,1] #Isolo primo punto dello spigolo
		v2 = linepoints[:,2] #Isolo secondo punto dello spigolo
		points = [ v1 + t*(v2 - v1) for t in params]   # !!!! loved !!
		#Creo un array che conterrà gli id dei punti d'intersezione trovati (verticispigolo)
		vs = zeros(Int64,1,length(points))
		PRECISION = 8
		numpoint = length(points)
		# Per ogni punto d'intersezione trovato sullo spigolo....
		@inbounds @simd for h = 1:numpoint
			#Approssimo coordinate del punto(x,y) trovato di un epsilon 
			points[h] = map(approxVal(PRECISION), points[h])
			#Se non ho mai visto prima il punto....
			if !haskey(vertdict, points[h])
				k += 1 #Genero ID punto 
				vertdict[points[h]] = k #Associo l'ID al punto
				push!(W, points[h]) #Pusho il punto(x,y) nell'array W
			end
			vs[h] = vertdict[points[h]] 
			#Assegno l'id del punto trovato nell'array dei punti d'intersezione
		end
		m = length(vs) - 1
		#se ho N punti d'intersezione trovati, genero N-1 spigoli 
		#ESEMPIO: se vs=[34,35,36,37] vs[h=1]=34, vs[h=2]=35, vs[h=3]=36, vs[h=4]=37
		# allora andrò a creare le coppie [34,35],[35,36],[36,37] come 3 spigoli. Queste coppie 		le pusho in EW
		@inbounds @simd for k=1:m
			push!(EW, [vs[k], vs[k+1]])
		end
	end
	W,EW = hcat(W...),convert(Array{Array{Int64,1},1},EW)
	V,EV = congruence((W,EW))
	return V,EV
end

function congruence(model)
	W,EW = model
	n = size(W,2)
	#Inizializzo un BallTree (albero che suddivide l'insieme dei punti in sfere)
	balltree = NearestNeighbors.BallTree(W)
	#Inizializzo raggio di ricerca
	r = 0.0000000001
	#Inizializzo un array vuoto di W elementi (W è una matrice 2xW )
	near = Array{Any}(undef, n)
	#Per ogni vertice...
	@inbounds @threads for k=1:n
		#Cerco i vertici più vicini nel raggio R=0.00000000001
		near[k] = NearestNeighbors.inrange(balltree, W[:,k], r, true)
	end
	#Ordino ogni array (near è bidimensionale)
	near = map(sort,near)
	#Cambio ogni vertice(x,y), col suo vertice(x,y) più vicino 
	@inbounds @threads for k=1:n
		W[:,k] = W[:,near[k][1]]
	end
	#Calcolo nuovi ID dei vertici
	pointidx = Array{Int64}(undef, n)
	@inbounds @threads for k=1:n
		pointidx[k] = near[k][1] 
	end
	l = length(pointidx)
	#Creo dict di trasformazione key(1..N) -> value(nuovoId[1],....,nuovoId[N]) 
	invidx = OrderedDict(zip(1:l, pointidx))
	V = Array{Array{Float64,1}}(undef, l)
	#Immagazzino W per righe (W è Nx2, V è 2xN)
	@inbounds @threads for k=1:l
		V[k] = W[:,k] 
	end
	#Creo EV aggiornato
	EV = []
	m = length(EW)
	#Per ogni spigolo in EW..
	@inbounds for i = 1:m
		#Rietichetto i vertici dello spigolo col dict di trasformazione
		newedge = [invidx[EW[i][1]],invidx[EW[i][2]]]
		#Elimino spigoli [2,2] che non esistono (2 === 2, 2 !== 2.0)
		if newedge[1] !== newedge[2]
			push!(EV,newedge)
		end
	end
	filter!(x ->  length(x)==2, EV)
	EV = convert(Lar.Cells, EV)
	return hcat(V...),EV
end

function approxVal(PRECISION)
	function approxVal0(value)
		out = round(value, digits=PRECISION)
		if out == -0.0
			out = 0.0
		end
		return out
	end
	return approxVal0
	end











function skel_merge(V1::Lar.Points, EV1::Lar.ChainOp, V2::Lar.Points, EV2::Lar.ChainOp)
    V = [V1; V2]
    EV = blockdiag(EV1,EV2)
    return V, EV
end

function submanifold_mapping(vs)
    u1 = vs[2,:] - vs[1,:]
    u2 = vs[3,:] - vs[1,:]
    u3 = cross(u1, u2)
    T = Matrix{Float64}(LinearAlgebra.I, 4, 4)
    T[4, 1:3] = - vs[1,:]
    M = Matrix{Float64}(LinearAlgebra.I, 4, 4)
    M[1:3, 1:3] = [u1 u2 u3]
    return T*M
end

function frag_face(V, EV, FE, sp_idx, sigma)

    vs_num = size(V, 1)

	# 2D transformation of sigma face
    sigmavs = (abs.(FE[sigma:sigma,:]) * abs.(EV))[1,:].nzind
    sV = V[sigmavs, :]
    sEV = EV[FE[sigma, :].nzind, sigmavs]
    M = submanifold_mapping(sV)
    sV = ([sV ones(size(sV,1))]*M)[:, 1:3] 
    # sigma face intersection with faces in sp_idx[sigma]
    for i in sp_idx[sigma]
	faceivs = (abs.(FE[i:i,:]) * abs.(EV))[1,:].nzind
	faceiV = V[faceivs, :]
	tV = ([faceiV ones(size(faceiV, 1))]*M)[:, 1:3]  
        tmpV, tmpEV = face_int(tV, EV)
        sV, sEV = skel_merge(sV, sEV, tmpV, tmpEV)
    end
    
    # computation of 2D arrangement of sigma face
    sV = sV[:, 1:2]
    nV, nEV, nFE = Lar.Arrangement.planar_arrangement(sV, sEV, sparsevec(ones(Int8, length(sigmavs))))
    if nV == nothing ## not possible !! ... (each original face maps to its decomposition)
        return [], spzeros(Int8, 0,0), spzeros(Int8, 0,0)
    end
    nvsize = size(nV, 1)
    nV = [nV zeros(nvsize) ones(nvsize)]*inv(M)[:, 1:3] ## ????
    return nV, nEV, nFE
    
end

function face_int(V::Lar.Points, EV::Lar.ChainOp)
    retV = Lar.Points(undef, 0, 3)
    visited_verts = []
    for i in 1:size(V,1)
        o = V[i,:]
        j = i < size(V,1) ? i+1 : 1
        d = V[j,:] - o
        err = 10e-8
        # err = 10e-4
        if !(-err < d[3] < err)

            alpha = -o[3] / d[3]

            if -err <= alpha <= 1+err
                p = o + alpha*d

                if -err < alpha < err || 1-err < alpha < 1+err
                    if !(Lar.vin(p, visited_verts))
                        push!(visited_verts, p)
                        retV = [retV; reshape(p, 1, 3)]
                    end
                else
                    retV = [retV; reshape(p, 1, 3)]
                end
            end
        end

    end

    vnum = size(retV, 1)


    if vnum == 1
        vnum = 0
        retV = Lar.Points(undef, 0, 3)
    end
    enum = (÷)(vnum, 2)
    retEV = spzeros(Int8, enum, vnum)

    for i in 1:enum
        retEV[i, 2*i-1:2*i] = [-1, 1]
    end

    retV, retEV
end

function planar_arrangement_1( V,copEV,sigma::Lar.Chain=spzeros(Int8,0),
                               return_edge_map::Bool=false,multiproc::Bool=false)
	# data structures initialization
	edgenum = size(copEV, 1)
	edge_map = Array{Array{Int, 1}, 1}(undef,edgenum)
	rV = Lar.Points(zeros(0, 2))
	rEV = SparseArrays.spzeros(Int8, 0, 0)
	finalcells_num = 0

	# spaceindex computation
	model = (convert(Lar.Points,V'),Lar.cop2lar(copEV))
	bigPI = spaceindex(model::Lar.LAR)

    
        # sequential (iterative) processing of edge fragmentation
        for i in 1:edgenum
            v, ev = frag_edge(V, copEV, i, bigPI)
            newedges_nums = map(x->x+finalcells_num, collect(1:size(ev, 1)))
            edge_map[i] = newedges_nums
            finalcells_num += size(ev, 1)
            rV = convert(Lar.Points, rV)
            rV, rEV = skel_merge(rV, rEV, v, ev)
        end
    
    # merging of close vertices and edges (2D congruence)
    V, copEV = rV, rEV
    V, copEV = merge_vertices!(V, copEV, edge_map)
	return V,copEV,sigma,edge_map
end

function frag_edge(V, EV::Lar.ChainOp, edge_idx::Int, bigPI)
    #Thread safety data structures    
    nth = nthreads()
    lbp = length(bigPI[edge_idx])
    alphaT=Array{Array{Float64}}(undef, lbp)
    vertsT = Array{Array{Float64,2}}(undef, nth)
    for i=1:nth
         vertsT[i] = Array{Float64,2}(undef,0,2)
    end
    edge = EV[edge_idx, :]
    @threads for it=1:lbp
        alphaT[it] = Array{Float64}(undef,0)
        tid = threadid() #Thread associato all'iterazione corrente it
        i=bigPI[edge_idx][it] #Edge da intersecare
        if i != edge_idx
            intersection = intersect_edges(V, edge, EV[i, :])
            for (point, alpha) in intersection
                vertsT[tid] = [vertsT[tid]; point]
                push!(alphaT[it],alpha) 
            end
        end
    end
    #Inizializzo strutture da ritornare
    verts = V[edge.nzind, :]
    for i=1:nth
        verts = [verts; vertsT[i]]
    end
    alphas = Dict{Float64, Int}()
    n=3
    for it=1:length(alphaT)
        for alpha in alphaT[it]
            alphas[alpha] = n
            n=n+1
        end
    end
    alphas[0.0], alphas[1.0] = [1, 2]
    alphas_keys = sort(collect(keys(alphas)))
    edge_num = length(alphas_keys)-1
    verts_num = size(verts, 1)
    ev = SparseArrays.spzeros(Int8, edge_num, verts_num)
    for i in 1:edge_num
        ev[i, alphas[alphas_keys[i]]] = 1
        ev[i, alphas[alphas_keys[i+1]]] = 1
    end
    return verts, ev
end

"""
    intersect_edges(V::Lar.Points, edge1::Lar.Cell, edge2::Lar.Cell)
Intersect two 2D edges (`edge1` and `edge2`).
"""
function intersect_edges(V::Lar.Points, edge1::Lar.Cell, edge2::Lar.Cell)
    err = 10e-8

    x1, y1, x2, y2 = vcat(map(c->V[c, :], edge1.nzind)...)
    x3, y3, x4, y4 = vcat(map(c->V[c, :], edge2.nzind)...)
    ret = Array{Tuple{Lar.Points, Float64}, 1}()

    v1 = [x2-x1, y2-y1];
    v2 = [x4-x3, y4-y3];
    v3 = [x3-x1, y3-y1];
    ang1 = dot(normalize(v1), normalize(v2))
    ang2 = dot(normalize(v1), normalize(v3))
    parallel = 1-err < abs(ang1) < 1+err
    colinear = parallel && (1-err < abs(ang2) < 1+err || -err < norm(v3) < err)
    if colinear
        o = [x1 y1]
        v = [x2 y2] - o
        alpha = 1/dot(v,v')
        ps = [x3 y3; x4 y4]
        for i in 1:2
            a = alpha*dot(v',(reshape(ps[i, :], 1, 2)-o))
            if 0 < a < 1
                push!(ret, (ps[i:i, :], a))
            end
        end
    elseif !parallel
        denom = (v2[2])*(v1[1]) - (v2[1])*(v1[2])
        a = ((v2[1])*(-v3[2]) - (v2[2])*(-v3[1])) / denom
        b = ((v1[1])*(-v3[2]) - (v1[2])*(-v3[1])) / denom

        if -err < a < 1+err && -err <= b <= 1+err
            p = [(x1 + a*(x2-x1))  (y1 + a*(y2-y1))]
            push!(ret, (p, a))
        end
    end
    return ret
end


function merge_vertices!(V::Lar.Points, EV::Lar.ChainOp, edge_map, err=1e-4)
    vertsnum = size(V, 1)
    edgenum = size(EV, 1)
    newverts = zeros(Int, vertsnum)
    # KDTree constructor needs an explicit array of Float64
    V = Array{Float64,2}(V)
    kdtree = KDTree(permutedims(V))

    # merge congruent vertices
    todelete = []
    i = 1
    for vi in 1:vertsnum
        if !(vi in todelete)
            nearvs = Lar.inrange(kdtree, V[vi, :], err)
            newverts[nearvs] .= i
            nearvs = setdiff(nearvs, vi)
            todelete = union(todelete, nearvs)
            i = i + 1
        end
    end
    nV = V[setdiff(collect(1:vertsnum), todelete), :]

    # merge congruent edges
    edges = Array{Tuple{Int, Int}, 1}(undef, edgenum)
    oedges = Array{Tuple{Int, Int}, 1}(undef, edgenum)
    @threads for ei=1:edgenum
        v1, v2 = EV[ei, :].nzind
        edges[ei]  = Tuple{Int, Int}(newverts[v1]<newverts[v2] ? [newverts[v1], newverts[v2]] : [newverts[v2], newverts[v1]])
        oedges[ei] = Tuple{Int, Int}(v1<v2 ? [v1, v2] :  [v2, v1])
    end
    nedges = union(edges)
    nedges = filter(t->t[1]!=t[2], nedges)
    nedgenum = length(nedges)
    nEV = spzeros(Int8, nedgenum, size(nV, 1))
    # maps pairs of vertex indices to edge index
    etuple2idx = Dict{Tuple{Int, Int}, Int}()
    # builds `edge_map`
    for ei in 1:nedgenum
        nEV[ei, collect(nedges[ei])] .= 1
        etuple2idx[nedges[ei]] = ei
    end
    @threads for i=1:length(edge_map)
        rowT=Array{Tuple{Int64,Int64}}(undef,length(edge_map[i]))
        len = length(edge_map[i])
        for j=1:len
            rowT[j]=edges[edge_map[i][j]]
        end
        filter!(t->t[1]!=t[2], rowT)
        edge_map[i]=Array{Int64}(undef,length(rowT))
        len2 = length(rowT)
        for j=1:len2
            edge_map[i][j]=etuple2idx[rowT[j]]
        end        
    end
    # return new vertices and new edges
    return Lar.Points(nV), nEV
end

function biconnected_components(EV::Lar.ChainOp)

    ps = Array{Tuple{Int, Int, Int}, 1}()
    es = Array{Tuple{Int, Int}, 1}()
    todel = Array{Int, 1}()
    visited = Array{Int, 1}()
    bicon_comps = Array{Array{Int, 1}, 1}()
    hivtx = 1

    function an_edge(point) # TODO: fix bug
        # error? : BoundsError: attempt to access 0×0 SparseMatrix ...
        edges = setdiff(EV[:, point].nzind, todel)
        if length(edges) == 0
            edges = [false]
        end
        edges[1]
    end

    function get_head(edge, tail)
        setdiff(EV[edge, :].nzind, [tail])[1]
    end

    function v_to_vi(v)
        i = findfirst(t->t[1]==v, ps)
        # seems findfirst changed from 0 to Nothing
        if typeof(i) == Nothing
            return false
        elseif i == 0
            return false
        else
            return ps[i][2]
        end
    end

    push!(ps, (1,1,1))
    push!(visited, 1)
    exit = false
    while !exit
        edge = an_edge(ps[end][1])
        if edge != false
            tail = ps[end][2]
            head = get_head(edge, ps[end][1])
            hi = v_to_vi(head)
            if hi == false
                hivtx += 1
                push!(ps, (head, hivtx, ps[end][2]))
                push!(visited, head)
            else
                if hi < ps[end][3]
                    ps[end] = (ps[end][1], ps[end][2], hi)
                end
            end
            push!(es, (edge, tail))
            push!(todel, edge)
        else
            if length(ps) == 1
                found = false
                pop!(ps)
                for i in 1:size(EV,2)
                    if !(i in visited)
                        hivtx = 1
                        push!(ps, (i, hivtx, 1))
                        push!(visited, i)
                        found = true
                        break
                    end
                end
                if !found
                    exit = true
                end

            else
                if ps[end][3] == ps[end-1][2]
                    edges = Array{Int, 1}()
                    while true
                        edge, tail = pop!(es)
                        push!(edges, edge)
                        if tail == ps[end][3]
                            if length(edges) > 1
                                push!(bicon_comps, edges)
                            end
                            break
                        end
                    end

                else
                    if ps[end-1][3] > ps[end][3]
                        ps[end-1] = (ps[end-1][1], ps[end-1][2], ps[end][3])
                    end
                end
                pop!(ps)
            end
        end
    end
    bicon_comps = sort(bicon_comps, lt=(x,y)->length(x)>length(y))
    return bicon_comps
end

function DFV_visit( VV::Lar.Cells, out::Array, count::Int, visited::Array, parent::Array, d::Array, low::Array, stack::Array, u::Int )::Array
		
    visited[u] = true
    count += 1
    d[u] = count
    low[u] = d[u]
    for v in VV[u]
        if ! visited[v]
            push!(stack, [(u,v)])
            parent[v] = u
            DFV_visit( VV,out,count,visited,parent,d,low,stack, v )
            if low[v] >= d[u]
                push!(out, [outputComp(stack,u,v)])
            end
            low[u] = min( low[u], low[v] )
        else
            if ! (parent[u]==v) && (d[v] < d[u])
                push!(stack, [(u,v)])
            end
            low[u] = min( low[u], d[v] )
        end
    end
    out
end

function outputComp(stack::Array, u::Int, v::Int)::Array
    out = []
    while true
        e = pop!(stack)[1]
        push!(out,e)
        if e == (u,v) 
        	break
        end
    end
    return [out] 
end

function biconnectedComponent(model)
    W,EV = model
    V = collect(1:size(W,2))
    count = 0
    stack,out = [],[]
    visited = [false for v in V]
    parent = Union{Int, Array{Any,1}}[[] for v in V]
    d = Any[0 for v in V]
    low = Any[0 for v in V]    
    VV = Lar.verts2verts(EV)
    out = Any[]
    for u in V 
        if ! visited[u] 
            out = DFV_visit( VV,out,count,visited,parent,d,low,stack, u )
        end
    end
    out = [component for component in out if length(component) >= 1]
    EVs = [map(sort∘collect,edges) for edges in cat((out...)...,dims = 1) if length(edges)>1] 
    EVs = filter(x->!isempty(x), EVs)
    bico = map(x ->sort(collect(Set(hcat(x...)))), EVs)
    return bico
end

end # module
