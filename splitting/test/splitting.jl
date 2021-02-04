import splitting
using Test
	@testset "greet" begin
		@test splitting.greet() == "ciao"
	end
