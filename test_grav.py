from graphviz import Source

src = Source('digraph graphname{rankdir=TB;140664582829072 [shape="octagon",style="filled",fillcolor="#E0E0E0",label="(2), float32"];140664582828624 [shape="record",style="filled",fillcolor="#6495ED",label="LinearFunction"];140664582724240 [shape="octagon",style="filled",fillcolor="#E0E0E0",label="(2, 3), float32"];140664582829264 [shape="octagon",style="filled",fillcolor="#E0E0E0",label="(2), float32"];140664582828752 [shape="octagon",style="filled",fillcolor="#E0E0E0",label="b: (2), float32"];140664582828304 [shape="record",style="filled",fillcolor="#6495ED",label="GetItem"];140664582828880 [shape="octagon",style="filled",fillcolor="#E0E0E0",label="(2, 2), float32"];140664582828944 [shape="record",style="filled",fillcolor="#6495ED",label="GetItem"];140664582828496 [shape="octagon",style="filled",fillcolor="#E0E0E0",label="W: (2, 3), float32"];140664582828880 -> 140664582828944;140664582828304 -> 140664582829072;140664582828752 -> 140664582828624;140664582828944 -> 140664582829264;140664582828624 -> 140664582828880;140664582828880 -> 140664582828304;140664582724240 -> 140664582828624;140664582828496 -> 140664582828624;}')

src.render('./test-output.gv', view=True)