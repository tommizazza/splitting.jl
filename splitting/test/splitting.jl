import splitting
using Test
using OrderedCollections
using BenchmarkTools
using LinearAlgebraicRepresentation
using IntervalTrees
Lar = LinearAlgebraicRepresentation

	@testset "boxcovering" begin
		V = [0.0 1.0 0.0 0.0 -1.0 -1.0; 
		     0.0 0.0 1.0 0.5 0.0 0.5]
		CV = [[1,2],[1,3],[2,3],[4,5],[4,6],[5,6]]
		cellpoints = [ V[:,CV[k]]::Lar.Points for k=1:length(CV) ]
		#----------------------------------------------------------
		bboxes = [hcat(splitting.boundingbox(cell)...) for cell in cellpoints]
		xboxdict = splitting.coordintervals(1,bboxes)
		yboxdict = splitting.coordintervals(2,bboxes)
		# xs,ys are IntervalTree type
		xs = IntervalTrees.IntervalMap{Float64, Array}()
		for (key,boxset) in xboxdict
			xs[tuple(key...)] = boxset
		end
		ys = IntervalTrees.IntervalMap{Float64, Array}()
		for (key,boxset) in yboxdict
			ys[tuple(key...)] = boxset
		end
		@test splitting.boxcovering(bboxes, 1, xs) == [[4, 5, 2, 1, 3], [4, 5, 2, 1, 3], [4, 5, 2, 1, 3], [6, 4, 5, 2, 1, 3], [6, 4, 5, 2, 1, 3], [6, 4, 5]]

		V1 = [0.0 1.0; 
		     0.0 0.0]
		EV1 = [[1,2]]
		cellpoints = [ V1[:,EV1[k]]::Lar.Points for k=1:length(EV1) ]
		#----------------------------------------------------------
		bboxes = [hcat(splitting.boundingbox(cell)...) for cell in cellpoints]
		xboxdict = splitting.coordintervals(1,bboxes)
		yboxdict = splitting.coordintervals(2,bboxes)
		# xs,ys are IntervalTree type
		xs = IntervalTrees.IntervalMap{Float64, Array}()
		for (key,boxset) in xboxdict
			xs[tuple(key...)] = boxset
		end
		ys = IntervalTrees.IntervalMap{Float64, Array}()
		for (key,boxset) in yboxdict
			ys[tuple(key...)] = boxset
		end
		@test splitting.boxcovering(bboxes, 1, xs) == [[1]]
	end

	@testset "coordintervals" begin
		@test splitting.coordintervals(1, []) == OrderedDict{Array{Float64,1},Array{Int64,1}}() 
		@test splitting.coordintervals(1, [[0.0 0.0; 0.0 0.0]]) == OrderedCollections.OrderedDict([0.0, 0.0] => [1])
		@test splitting.coordintervals(2, [[0.0 0.0; 0.0 0.0]]) == OrderedCollections.OrderedDict([0.0, 0.0] => [1])
		@test splitting.coordintervals(1, [[0.0 1.0; 0.0 0.0], [0.0 0.0;0.0 0.1]]) == OrderedDict([0.0, 1.0] => [1],[0.0, 0.0] => [2])
 		end

	@testset "spaceIndex" begin
		V = [0.0 1.0 0.0 0.0 -1.0 -1.0; 
		     0.0 0.0 1.0 0.5 0.0 0.5]
		EV = [[1,2],[1,3],[2,3],[4,5],[4,6],[5,6]]
		FV = [[1,2,3],[4,5,6]]
		@test splitting.spaceindex((V,EV)) == [[4,2,3],[4,5,1,3],[4,5,2,1],[6,5,2,1,3],[6,4,2,3],[4,5]]

		V1 = [0.0 2.0 2.0 -2.0 4.0 4.0 6.0 2.0 -1.0;
		      0.0 2.0 0.0 2.0 4.0 6.0 6.0 6.0 3.0]
		EV1 = [[1,2], [3,4], [5,6], [7,6], [8,9]]
		@test splitting.spaceindex((V1, EV1)) == [[2], [1], [4], [3], Int64[]]
	end

	@testset "linefragments" begin
		V = [0.0 1.0 0.0 1.0; 
		     0.0 1.0 1.0 0.0]
		EV = [[1,2],[3,4]]
		@test splitting.linefragments(V,EV, splitting.spaceindex((V, EV))) == [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]
		V1 = [0.0 1.0 ; 
		     0.0 0.0]
		EV1 = [[1,2]]
		@test splitting.linefragments(V1,EV1, splitting.spaceindex((V1, EV1))) == [[0.0, 1.0]]
	end

	@testset "intersection" begin
		#linee parallele
		line1 = [0.0 0.0; 0.0 1.0]
		line2 = [1.0 1.0; 0.0 1.0]
		@test splitting.intersection(line1, line2) == ()
		#linee intersecanti come una x
		line1 = [0.0 1.0; 0.0 1.0]
		line2 = [1.0 0.0; 0.0 1.0]
		@test splitting.intersection(line1, line2) == (0.5, 0.5)
		
	end

	@testset "fragmentlines" begin
		V = [0.0 1.0 0.0 1.0; 
		     0.0 1.0 1.0 0.0]
		EV = [[1,2],[3,4]]
		a, b =  splitting.fragmentlines((V,EV)) 
		@test a == [0.0 0.5 1.0 0.0 1.0; 0.0 0.5 1.0 1.0 0.0]
		@test b == [[1, 2], [2, 3], [4, 2], [2, 5]]
		V1 = [0.0 1.0 ; 
		     0.0 0.0]
		EV1 = [[1,2]]
		a, b = splitting.fragmentlines((V1,EV1))
		@test V1 == a
		@test EV1 == b
		
	end

	@testset "congruence" begin
		V1 = [0.0 1.0 ; 
		     0.0 0.0]
		EV1 = [[1,2]]
		a, b = splitting.congruence((V1,EV1))
		@test a == V1
		@test b == EV1

		V1 = [0.0 1.0 0.99999999999999999 ; 
		     0.0 0.0 0.0]
		EV1 = [[1,2], [1,3]]
		a, b = splitting.congruence((V1,EV1))
		@test b == [[1, 2], [1, 2]] 
		
	end
	
