BAR = 4*60/87
def ts(t): return f"{int(t//60)}:{t%60:05.2f}"

HOOK = ["svake burne noci ne mogu da spavam",
        "a mozda je i bolje da se ne nadam",
        "a ja zbog tebe zivim ko u staklu",
        "al to je u redu jer volis me takvu",
        "volis me takvu  (tag)"]
V1 = """od malena mi u glavu tuku vazne teme
govore mi da sa tobom samo tracim vreme
krecu prve suze ne znam sta da mislim
da nekog drugog volis mozda si sa bivsim
ma nemoguce to ne moze meni da se desi
jedno smo za drugo oboje smo svesni
a onda meni nada jedina preostaje
za svaku moju gresku sto treba da se spoznaje
ali ovo nase nikako da ispari
ostace u srcu cak i kad se ostari
ti nekada najglasnija sada tiha i bleda
ovo polako pocinje da mi smeta
u hodniku je rupa jos od one zime
niko je ne zatvara navikli smo s time
kod nas se tako radi o tome se ne prica
iza zatvorenih vrata ista je ta slika""".split("\n")
V2 = """a onda dolazi moja strana price
kao malo seme sto tek krece da nice
od malih nogu se znamo ali ovo moram reci
i ovo razdoblje trebalo je proci
prevare i lazi ma nek prestanu bre vise
iako mome srcu nikad nisi bila blize
ali evo bacam sada ove rime
pokusavam da razumem sta nam sve ne ide
ali ova prica ne moze da stane
necu dati svojoj dusi da mi klone
ne mogu da podnesem ne mogu da shvatim
od toliko ljudi sta te je navelo da svratis
u moj zivot i promenis me celog
da si videla bar lepog a ne mene vrednog
znam da strepis znam da bezis mogu sve da vidim
i dalje nista ne menjam samo dalje krivim""".split("\n")
V3 = """ne pustam da odes ne pustam da dises
sa ovog voza ne mozes da sidjes
rascistimo ovo jednom i za svagda
kao nasa prva svadja nek i ovo tako strada
volim te priznajem nikada negiro
zao mi je sto se sve ovako desilo
prikrada se polako kajanje me spopada
mozda su tako skovana deca belog grada
odrastali i druzili samo se provodili
dok smo bili zajedno opet smo se rodili
i pored kraja ostao je ukus tog istog grada
i dalje pamtim sve sto si mi rekla tada
zaboravi na psovke nasilje i vike
a meni fale jutra kada trazim izlike
i za kraj ovog sumornoga dana
vrata su zakljucana ti nemas kuda sama""".split("\n")

out=[]
out.append("VOLIS ME TAKVU - lyric timing sheet")
out.append("87 BPM, 4/4, E minor.  1 bar = 2.759 s.  88 bars total.\n")
def hook(n, bar):
    out.append(f"--- HOOK {n}  (bars {bar}-{bar+7})  {ts(bar*BAR)} - {ts((bar+8)*BAR)}  [2 bars per line] ---")
    for i,l in enumerate(HOOK[:4]):
        out.append(f"  {ts((bar+i*2)*BAR)}  bar {bar+i*2:<3} {l}")
    out.append(f"  {ts((bar+7)*BAR)}  bar {bar+7:<3} {HOOK[4]}")
    out.append("")
def verse(n, bar, lines):
    out.append(f"--- VERSE {n}  (bars {bar}-{bar+15})  {ts(bar*BAR)} - {ts((bar+16)*BAR)}  [1 bar per line] ---")
    for i,l in enumerate(lines):
        out.append(f"  {ts((bar+i)*BAR)}  bar {bar+i:<3} {l}")
    out.append("")

out.append(f"--- INTRO  (bars 0-3)  {ts(0)} - {ts(4*BAR)}  piano alone, count in ---\n")
hook(1,4); verse(1,12,V1); hook(2,28); verse(2,36,V2)
hook(3,52); verse(3,60,V3); hook(4,76)
out.append(f"--- OUTRO  (bars 84-87)  {ts(84*BAR)} - {ts(88*BAR)}  instruments fall away, ends solo piano ---")
open("output/volis-me-takvu-timing.txt","w").write("\n".join(out)+"\n")
print("\n".join(out[:26]))
print("...")
print(f"\nwritten: output/volis-me-takvu-timing.txt ({len(out)} lines)")
