import sys, glob, collections
# Survey bundled ffields using the same line layout as reaxff_ffield.cpp (read-only analysis, not an engine component).
def survey(path):
    L = [l.rstrip('\n') for l in open(path, errors='replace')]
    i = 1
    ng = int(L[i].split()[0]); i += 1
    gp = [float(L[i+k].split()[0]) for k in range(ng)]; i += ng
    nat = int(L[i].split()[0]); i += 4
    # detect 4-line vs 5-line atom blocks by whether the 5th line starts a new element symbol
    atoms = []
    lines_per = 4
    first = L[i].split()
    # try 4 first: line i+4 should start with alphabetic symbol (next atom) or be a bond-count integer
    nxt = L[i+4].split()
    try:
        float(nxt[0]); is_alpha = False
    except ValueError:
        is_alpha = True
    if nat > 1 and not is_alpha: lines_per = 5
    elif nat == 1:
        lines_per = 5 if len(L[i+4].split())==2 and L[i+4].split()[0].replace('.','',1).replace('-','',1).isdigit() and 'bonds' not in L[i+4].lower() else 4
    for a in range(nat):
        b = L[i+a*lines_per:i+(a+1)*lines_per]
        w = b[0].split()
        atoms.append(dict(sym=w[0].upper(), mass=float(w[3]), rcore2=float(b[3].split()[5]), acore2=float(b[3].split()[7]), gamma_w=float(b[1].split()[1]), phb=int(float(b[1].split()[7])), valency=float(w[2]), valboc=float(b[1].split()[2]), valval=float(b[3].split()[3])))
    i += nat*lines_per
    nbd = int(L[i].split()[0]); i += 2 + 2*nbd
    noff = int(L[i].split()[0]); i += 1 + noff
    nang = int(L[i].split()[0]); i += 1
    trip = collections.Counter(); sym_self = 0
    for k in range(nang):
        w = L[i+k].split(); j,kk,l = int(w[0]),int(w[1]),int(w[2])
        trip[(j,kk,l)] += 1
        if j == l: sym_self += 1
    dup = sum(1 for k,v in trip.items() if v>1 or (k[::-1] in trip and k[::-1]!=k))
    i += nang
    ntor = int(L[i].split()[0]); i += 1
    compact = sum(1 for k in range(ntor) if int(L[i+k].split()[0])==0 or int(L[i+k].split()[3])==0)
    i += ntor
    nhb = int(L[i].split()[0]) if i < len(L) and L[i].split() else 0
    vdw = set()
    for a in atoms:
        inner = a['rcore2']>0.01 and a['acore2']>0.01; shield = a['gamma_w']>0.5
        vdw.add(3 if inner and shield else 2 if inner else 1 if shield else -1)
    return dict(file=path.split('/')[-1], nglobal=ng, nat=nat, per=lines_per, elems=[a['sym'] for a in atoms],
                heavy=[a['sym'] for a in atoms if a['mass']>=21], hb=[a['sym'] for a in atoms if a['phb']>0],
                nang=nang, ang_jeql_self=sym_self, ang_dup_or_mirror=dup, ntor=ntor, tor_compact=compact, nhb=nhb, vdw=sorted(vdw),
                valval_ne_valboc_light=[a['sym'] for a in atoms if a['mass']<21 and a['valval']!=a['valboc']], gp37=gp[37] if ng>37 else None, gp35=gp[35] if ng>35 else None)
for p in sorted(glob.glob(sys.argv[1]+'/ffield.reax.*')):
    try:
        r = survey(p)
        print(r['file'], '| nat', r['nat'], 'lines/atom', r['per'], r['elems'], '| heavy(m>=21):', r['heavy'], '| hb-types:', r['hb'])
        print('    3b:', r['nang'], 'j==l:', r['ang_jeql_self'], 'dup/mirror:', r['ang_dup_or_mirror'], '| 4b:', r['ntor'], 'compact0-X-Y-0:', r['tor_compact'], '| hbond rows:', r['nhb'], '| vdw_type(s):', r['vdw'], '| gp37:', r['gp37'], '| light valval!=valboc:', r['valval_ne_valboc_light'])
    except Exception as e:
        print(p.split('/')[-1], 'SURVEY-FAILED:', type(e).__name__, e)
