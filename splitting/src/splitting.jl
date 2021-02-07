module splitting

	using LinearAlgebraicRepresentation
	Lar = LinearAlgebraicRepresentation
	using IntervalTrees
	using SparseArrays
	using NearestNeighbors
	using DataStructures
	using OrderedCollections
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
	   minimum = mapslices(x->min(x...), vertices, dims=2)
	   maximum = mapslices(x->max(x...), vertices, dims=2)
	   return minimum, maximum
	end

	#Questa funzione computa un dizionario avente come chiave una coordinata [cmin,cmax], e
	#come valore un array contenente gli indici di tutte le celle incidenti sulla chiave. 
	function coordintervals(coord,bboxes)
		boxdict = OrderedDict{Array{Float64,1},Array{Int64,1}}()
		#Per ogni bounding box...
		for (h,box) in enumerate(bboxes)
			#La chiave del dizionario è [cmin,cmax]
			key = box[coord,:]
			#Se il dizionario non ha la chiave, creo la entry..
			if haskey(boxdict,key) == false
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
		covers = [[] for k=1:length(bboxes)]
		#Per ogni bbox....
		for (i,boundingbox) in enumerate(bboxes)
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
			if (y1==y2) == (y3==y4) || (x1==x2) == (x3==x4) # segments collinear
				 return ()
			else
				 # segments parallel: no intersection
				 return ()
			end
		end
		return α,β
	end


	function linefragments(V,EV,Sigma)
		# Inizializzo dati
		m = length(Sigma) #Numero segmenti
		sigma = map(sort,Sigma) #ordino i segmenti intersecati di ogni segmento ([[2,3,....],[1,3,.....])
		reducedsigma = sigma ##[filter(x->(x > k), sigma[k]) for k=1:m]
		# pairwise parametric intersection
		params = Array{Float64,1}[[] for i=1:m]
		#Per ogni segmento...
		for h=1:m
			#Se il segmento si interseca con qualcosa:
			if sigma[h] ≠ []
				#Prendo i punti(x1,y1),(x2,y2) del segmento h-esimo:
				line1 = V[:,EV[h]]
				#Confronto il segmento h-esimo con tutti gl altri segmenti
				#presenti nel suo indice spaziale.
				for k in sigma[h]
					#Prendo i punti(x3,y3),(x4,y4) del segmento k-esimo presente in Sigma[h]:
					line2 = V[:,EV[k]]
					#Ritorno (se esistono) i parametri (alfa,beta) necessari a calcolare
					#il punto di intersezione tra coppie di segmenti, h <-> k
					out = intersection(line1,line2) 
					#Se ho intersezione tra le rette (ovvero out non è nothing)
					if out ≠ ()
						#Controllo che i parametri α,β siano ammissibili, se lo sono
						#li immagazzino nella struttura dati params
						α,β = out
						if 0<=α<=1 && 0<=β<=1
							push!(params[h], α)
							push!(params[k], β)
						end
					end
				end
			end
		end
		# Inizializzo struttura da ritornare
		fragparams = []
		#Per ogni spigolo..
		for line in params
			#Metto i valori 0.0 e 1.0 nei parametri d'intersezione di ogni spigolo
			# (in quanto i punti di uno spigolo definiscono per definizione i parametri 0 e 1)
			push!(line, 0.0, 1.0)
			#Tolgo i doppioni			
			line = sort(collect(Set(line)))
			#pusho linea senza doppioni sulla struttura dati da ritornare
			push!(fragparams, line)
		end
		return fragparams
	end


	function fragmentlines(model)
		V,EV = model
		# Creo indice spaziale
		Sigma = spaceindex(model)
		# calcolo parametri d'intersezione degli spigoli
		lineparams = linefragments(V,EV,Sigma)
		# initialization of local data structures
		vertdict = OrderedDict{Array{Float64,1},Array{Int,1}}()
		pairs = collect(zip(lineparams, [V[:,e] for e in EV]))
		vertdict = OrderedDict{Array{Float64,1},Int}()
		#Inizializzo nuovi V, EV per aggiungere i nuovi vertici/spigoli dello splitting
		W = Array[]
		EW = Array[]
		k = 0
		# Ricostruisco i nuovi punti generati dall'intersezione tra spigoli
		# tramite i parametri d'intersezione
		# Per ogni spigolo...
		for (params,linepoints) in pairs
			v1 = linepoints[:,1] #Isolo primo punto dello spigolo
			v2 = linepoints[:,2] #Isolo secondo punto dello spigolo
			# Calcolo un array contenente tutti i punti d'intersezione sullo spigolo (tanti quanti
			# sono i parametri d'intersez)			
			points = [ v1 + t*(v2 - v1) for t in params]   # !!!! loved !!
			#Creo un array che conterrà gli id dei punti d'intersezione trovati (verticispigolo)
			vs = zeros(Int64,1,length(points))
			PRECISION = 8
			# Per ogni punto d'intersezione trovato sullo spigolo....
			for (h,point) in enumerate(points)
				#Approssimo coordinate del punto(x,y) trovato di un epsilon 
				point = map(approxVal(PRECISION), point)
				#Se non ho mai visto prima il punto....
				if haskey(vertdict, point) == false
					k += 1 #Genero ID punto 
					vertdict[point] = k #Associo l'ID al punto
					push!(W, point) #Pusho il punto(x,y) nell'array W
				end
				vs[h] = vertdict[point] #Assegno l'id del punto trovato nell'array dei punti d'intersezione
			end
			[push!(EW, [vs[k], vs[k+1]]) for k=1:length(vs)-1]
		end
		#se ho N punti d'intersezione trovati, genero N-1 spigoli 
		#ESEMPIO: se vs=[34,35,36,37] vs[h=1]=34, vs[h=2]=35, vs[h=3]=36, vs[h=4]=37
		# allora andrò a creare le coppie [34,35],[35,36],[36,37] come 3 spigoli. Queste coppie le pusho in EW
		W,EW = hcat(W...),convert(Array{Array{Int64,1},1},EW)
		V,EV = congruence((W,EW))
		return V,EV
	end
		
	function congruence(model)
		W,EW = model
		#Inizializzo un BallTree (albero che suddivide l'insieme dei punti in sfere)
		balltree = NearestNeighbors.BallTree(W)
		#Inizializzo raggio di ricerca
		r = 0.0000000001
		#Inizializzo un array vuoto di W elementi (W è una matrice 2xW )
		near = Array{Any}(undef, size(W,2))
		#Per ogni vertice...
		for k=1:size(W,2)
			#Cerco i vertici più vicini nel raggio R=0.00000000001
			near[k] = NearestNeighbors.inrange(balltree, W[:,k], r, true)
		end
		#Ordino ogni array (near è bidimensionale)
		near = map(sort,near) 
		#Cambio ogni vertice(x,y), col suo vertice(x,y) più vicino
		for k=1:size(W,2)
			W[:,k] = W[:,near[k][1]]
		end
		#Calcolo nuovi ID dei vertici
		pointidx = [ near[k][1] for k=1:size(W,2) ] 
		#Creo dict di trasformazione key(1..N) -> value(nuovoId[1],....,nuovoId[N]) 
		invidx = OrderedDict(zip(1:length(pointidx), pointidx))
		#Immagazzino W per righe (W è Nx2, V è 2xN)
		V = [W[:,k] for k=1:length(pointidx)]

		#Creo EV aggiornato
		EV = []
		#Per ogni spigolo in EW..
		for e in (EW)
			#Rietichetto i vertici dello spigolo col dict di trasformazione
			newedge = [invidx[e[1]],invidx[e[2]]]
			#Elimino spigoli [2,2] che non esistono (2 === 2, 2 !== 2.0)
			if newedge[1] !== newedge[2]
				push!(EV,newedge)
			end
		end
		EV = [EV[h] for h=1:length(EV) if length(EV[h])==2]
		EV = convert(Lar.Cells, EV)
		#W,EW = Lar.simplifyCells(V,EV)
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
end # module
