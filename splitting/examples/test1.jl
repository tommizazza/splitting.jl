using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation
import splitting

b=[[],[],[]]
EV=[[1,1]]
FV=[[1,1,1]]
CV=[[1,1,1,1]]

for i=1:1000
	push!(b[1],(0.0 + i*0.8))
	push!(b[2],(0.0 + i*0.8))
	push!(b[3],(0.0 + i*0.8))

	push!(b[1],(3.0 + i*0.8))
	push!(b[2],(0.0 + i*0.8))
	push!(b[3],(0.0 + i*0.8))

	push!(b[1],(0.0 + i*0.8))
	push!(b[2],(3.0 + i*0.8))
	push!(b[3],(0.0 + i*0.8))

	push!(b[1],(0.0 + i*0.8))
	push!(b[2],(0.0 + i*0.8))
	push!(b[3],(3.0 + i*0.8))

	push!(EV,[1+4*(i-1),2+4*(i-1)])
	push!(EV,[1+4*(i-1),3+4*(i-1)])
	push!(EV,[1+4*(i-1),4+4*(i-1)])
	push!(EV,[2+4*(i-1),3+4*(i-1)])
	push!(EV,[2+4*(i-1),4+4*(i-1)])
	push!(EV,[3+4*(i-1),4+4*(i-1)])

	push!(FV,[1+4*(i-1),2+4*(i-1),3+4*(i-1)])
	push!(FV,[1+4*(i-1),2+4*(i-1),4+4*(i-1)])
	push!(FV,[1+4*(i-1),3+4*(i-1),4+4*(i-1)])
	push!(FV,[2+4*(i-1),3+4*(i-1),4+4*(i-1)])

	push!(CV,[1+4*(i-1),2+4*(i-1),3+4*(i-1),4+4*(i-1)])
end
V = permutedims(reshape(hcat(b...), (length(b[1]), length(b))))
filter!(e->e!=[1,1],EV)
filter!(e->e!=[1,1,1],FV)
filter!(e->e!=[1,1,1,1],CV)

copEV = Lar.coboundary_0(EV::Lar.Cells)
copFE = Lar.coboundary_1(V, FV::Lar.Cells, EV::Lar.Cells)
V = convert(Array{Float64,2},V') 

V, copEV, copFE, copCF = splitting.space_arrangement(V,copEV,copFE)
