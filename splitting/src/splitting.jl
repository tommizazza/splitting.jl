module splitting

	using LinearAlgebraicRepresentation
	Lar = LinearAlgebraicRepresentation
	using IntervalTrees
	using SparseArrays
	using NearestNeighbors
	using DataStructures
	using OrderedCollections

	function spaceindex(model::Lar.LAR)::Array{Array{Int,1},1}
		V,CV = model[1:2]
		dim = size(V,1)
		cellpoints = [ V[:,CV[k]]::Lar.Points for k=1:length(CV) ]
		#----------------------------------------------------------
		bboxes = [hcat(boundingbox(cell)...) for cell in cellpoints]
		xboxdict = coordintervals(1,bboxes)
		yboxdict = coordintervals(2,bboxes)
		# xs,ys are IntervalTree type
		xs = IntervalTrees.IntervalMap{Float64, Array}()
		for (key,boxset) in xboxdict
			xs[tuple(key...)] = boxset
		end
		ys = IntervalTrees.IntervalMap{Float64, Array}()
		for (key,boxset) in yboxdict
			ys[tuple(key...)] = boxset
		end
		xcovers = boxcovering(bboxes, 1, xs)
		ycovers = boxcovering(bboxes, 2, ys)
		covers = [intersect(pair...) for pair in zip(xcovers,ycovers)]

		if dim == 3
			zboxdict = coordintervals(3,bboxes)
			zs = IntervalTrees.IntervalMap{Float64, Array}()
			for (key,boxset) in zboxdict
				zs[tuple(key...)] = boxset
			end
			zcovers = boxcovering(bboxes, 3, zs)
			covers = [intersect(pair...) for pair in zip(zcovers,covers)]
		end
		# remove each cell from its cover
		for k=1:length(covers)
			covers[k] = setdiff(covers[k],[k])
		end
		return covers
	end


	function boundingbox(vertices::Lar.Points)
	   minimum = mapslices(x->min(x...), vertices, dims=2)
	   maximum = mapslices(x->max(x...), vertices, dims=2)
	   return minimum, maximum
	end

	function coordintervals(coord,bboxes)
		boxdict = OrderedDict{Array{Float64,1},Array{Int64,1}}()
		for (h,box) in enumerate(bboxes)
			key = box[coord,:]
			if haskey(boxdict,key) == false
				boxdict[key] = [h]
			else
				push!(boxdict[key], h)
			end
		end
		return boxdict
	end

	function boxcovering(bboxes, index, tree)
		covers = [[] for k=1:length(bboxes)]
		for (i,boundingbox) in enumerate(bboxes)
			extent = bboxes[i][index,:]
			iterator = IntervalTrees.intersect(tree, tuple(extent...))
			for x in iterator
				append!(covers[i],x.value)
			end
		end
		return covers
	end

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
				 return nothing
			else
				 # segments parallel: no intersection
				 return nothing
			end
		end
		return α,β
	end


	function linefragments(V,EV,Sigma)
		# remove the double intersections by ordering Sigma
		m = length(Sigma)
		sigma = map(sort,Sigma)
		reducedsigma = sigma ##[filter(x->(x > k), sigma[k]) for k=1:m]
		# pairwise parametric intersection
		params = Array{Float64,1}[[] for i=1:m]
		for h=1:m
			if sigma[h] ≠ []
				line1 = V[:,EV[h]]
				for k in sigma[h]
					line2 = V[:,EV[k]]
					out = intersection(line1,line2) # TODO: w interval arithmetic
					if out ≠ nothing
						α,β = out
						if 0<=α<=1 && 0<=β<=1
							push!(params[h], α)
							push!(params[k], β)
						end
					end
				end
			end
		end
		# finalize parameters of fragmented lines
		fragparams = []
		for line in params
			push!(line, 0.0, 1.0)
			line = sort(collect(Set(line)))
			push!(fragparams, line)
		end
		return fragparams
	end


	function fragmentlines(model)
		V,EV = model
		# acceleration via spatial index computation
		Sigma = spaceindex(model)
		# actual parametric intersection of each line with the close ones
		lineparams = linefragments(V,EV,Sigma)
		# initialization of local data structures
		vertdict = OrderedDict{Array{Float64,1},Array{Int,1}}()
		pairs = collect(zip(lineparams, [V[:,e] for e in EV]))
		vertdict = OrderedDict{Array{Float64,1},Int}()
		W = Array[]
		EW = Array[]
		k = 0
		# generation of intersection points
		for (params,linepoints) in pairs
			v1 = linepoints[:,1]
			v2 = linepoints[:,2]
			points = [ v1 + t*(v2 - v1) for t in params]   # !!!! loved !!
			vs = zeros(Int64,1,length(points))
			PRECISION = 8
			# identification via dictionary of points
			for (h,point) in enumerate(points)
				point = map(approxVal(PRECISION), point)
				if haskey(vertdict, point) == false
					k += 1
					vertdict[point] = k
					push!(W, point)
				end
				vs[h] = vertdict[point]
			end
			[push!(EW, [vs[k], vs[k+1]]) for k=1:length(vs)-1]
		end
		# normalization of output
		W,EW = hcat(W...),convert(Array{Array{Int64,1},1},EW)
		V,EV = congruence((W,EW))
		return V,EV
	end
	function fraglines(sx::Float64=1.2,sy::Float64=1.2,sz::Float64=1.2)
		function fraglines0(model)
			V,EV = fragmentlines(model)

			W = zeros(Float64, size(V,1), 2*length(EV))
			EW = Array{Array{Int64,1},1}()
			for (k,(v1,v2)) in enumerate(EV)
				if size(V,1)==2
					x,y = (V[:,v1] + V[:,v2]) ./ 2
					scx,scy = x*sx, y*sy
					t = [scx-x, scy-y]
				elseif size(V,1)==3
					x,y,z = (V[:,v1] + V[:,v2]) ./ 2
					scx,scy,scz = x*sx, y*sy, z*sz
					t = [scx-x, scy-y, scz-z]
				end
				W[:,2*k-1] = V[:,v1] + t
				W[:,2*k] = V[:,v2] + t
				push!(EW, [2*k-1, 2*k])
			end
			return W,EW
		end
		return fraglines0
	end



	
	function congruence(model)
		W,EW = model
		# congruent vertices
		balltree = NearestNeighbors.BallTree(W)
		r = 0.0000000001
		near = Array{Any}(undef, size(W,2))
		for k=1:size(W,2)
			near[k] = cat([NearestNeighbors.inrange(balltree, W[:,k], r, true)])
		end
		near = map(sort,near)  # check !!!
		for k=1:size(W,2)
			W[:,k] = W[:,near[k][1]]
		end
		pointidx = [ near[k][1] for k=1:size(W,2) ]  # check !!
		invidx = OrderedDict(zip(1:length(pointidx), pointidx))
		V = [W[:,k] for k=1:length(pointidx)]
		# congruent edges
		EV = []
		for e in (EW)
			newedge = [invidx[e[1]],invidx[e[2]]]
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
