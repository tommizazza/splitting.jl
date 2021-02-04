import splitting
using Test
using OrderedCollections

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
		@test splitting.intersection(line1, line2) == nothing
		#linee intersecanti come una x
		line1 = [0.0 1.0; 0.0 1.0]
		line2 = [1.0 0.0; 0.0 1.0]
		@test splitting.intersection(line1, line2) == (0.5, 0.5)
		
	end

	#TODO
	@testset "congruence" begin
		
		
	end
	
