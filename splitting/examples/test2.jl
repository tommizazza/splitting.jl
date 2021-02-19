import splitting
using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation

b=[[],[]]
EV=[[1,1]]

for i=1:10000
           push!(b[1],(1.0 + i*2.0))
           push!(b[2],(1.0 + i*2.0))
           push!(b[1],(4.0 + i*2.0))
           push!(b[2],(1.0 + i*2.0))
           push!(b[1],(1.0 + i*2.0))
           push!(b[2],(4.0 + i*2.0))
           push!(b[1],(4.0 + i*2.0))
           push!(b[2],(4.0 + i*2.0))
           push!(EV,[1+4*(i-1),2+4*(i-1)])
           push!(EV,[1+4*(i-1),3+4*(i-1)])
           push!(EV,[2+4*(i-1),4+4*(i-1)])
           push!(EV,[3+4*(i-1),4+4*(i-1)])
end

V = permutedims(reshape(hcat(b...), (length(b[1]), length(b))))
filter!(e->e!=[1,1],EV)

results = splitting.fragmentlines((V, EV))
