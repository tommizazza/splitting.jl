module splitting

	using LinearAlgebraicRepresentation
	Lar = LinearAlgebraicRepresentation
	using IntervalTrees
	using SparseArrays
	using NearestNeighbors
	using DataStructures

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
end # module
