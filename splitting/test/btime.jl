import splitting
using BenchmarkTools
using IntervalTrees

#Input sezione space index
V = [0.0 2.0 2.0 -2.0 4.0 4.0 6.0 2.0 -1.0;
      0.0 2.0 0.0  2.0 4.0 6.0 6.0 6.0  3.0]
EV = [[1,2], [3,4], [5,6], [7,6], [8,9]]

println("-BTIME FUNZIONE BOUNDINGBOX:\n\t")
@btime splitting.boundingbox(V[:,EV[1]])

println("\n\n-BTIME FUNZIONE COORDINTERVALS:\n\t")
cellpoints = [ V[:,EV[k]] for k=1:length(EV) ]
bboxes = [hcat(splitting.boundingbox(cell)...) for cell in cellpoints]
@btime splitting.coordintervals(1,bboxes)

println("\n\n-BTIME FUNZIONE BOXCOVERING:\n\t")
xboxdict = splitting.coordintervals(1,bboxes)
xs = IntervalTrees.IntervalMap{Float64, Array}()
for (key,boxset) in xboxdict
	xs[tuple(key...)] = boxset
end
@btime splitting.boxcovering(bboxes, 1, xs)

println("\n\n-BTIME FUNZIONE SPACEINDEX:\n\t")
@btime splitting.spaceindex((V,EV))

#Input per intersection
println("\n\n-BTIME FUNZIONE INTERSECTION:\n\t")
line1 = [0.0 1.0; 0.0 1.0]
line2 = [1.0 0.0; 0.0 1.0]
@btime splitting.intersection(line1, line2) == (0.5, 0.5)

println("\n\n-BTIME FUNZIONE LINEFRAGMENTS:\n\t")
spaceindex = splitting.spaceindex((V,EV))
@btime splitting.linefragments(V,EV,spaceindex)

println("\n\n-BTIME FUNZIONE CONGRUENCE:\n\t")
@btime splitting.congruence((V,EV))

println("\n\n-BTIME FUNZIONE FRAGMENTLINES:\n\t")
@btime splitting.fragmentlines((V,EV))
