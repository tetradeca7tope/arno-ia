import numpy
import os.path
subdtype = numpy.dtype([
    ('mass', 'f4'), 
    ('len', 'i4'), 
    ('pos', ('f4', 3)),           # potential min pos
    ('vel', ('f4', 3)),   
    ('vdisp', 'f4'),  
    ('vcirc', 'f4'),  
    ('rcirc', 'f4'),  
    ('parent', 'i4'),             # parent structure
    ('massbytype', ('f4', 6)),  
    ('lenbytype', ('u4',6)), 
    ('unused', 'u4'),    
    ('groupid', 'u4'),            # group id
   ])

groupdtype = numpy.dtype([
    ('mass', 'f4'),      
    ('len', 'i4'), 
    ('pos', ('f4', 3)),         # 
    ('vel', ('f4', 3)), 
    ('nhalo', 'i4'),            # number of subhalos + 1(contamination)
    ('massbytype', ('f4', 6)),  
    ('lenbytype', ('u4',6)), 
   ])

pdtype = numpy.dtype([
    ('pos', ('f4', 3)), # position in comoving kpc/h
    ('vel', ('f4', 3)), # velocity in proper units (NOT in GADGET internal unit)
    ('mass', 'f4'),     # mass in 1e10 Msun/h
    ('id', 'u8'),       # unit id of the particle
    ('type', 'u1'),     # type 0 (gas) 1(dm) 4(star) 5(bh)
    ('SEDindex', 'i8'),  # to look up stellar band luminosity
    ('recfrac', 'f4'),   # stellar recycling fraction (ask wilkins)
    ])

# dtype used for blackhole properties in subhalo/
bhdtype = numpy.dtype([
    ('pos', ('f4', 3)),
    ('vel', ('f4', 3)),
    ('id', 'u8'),
    ('bhmass', 'f8'),
    ('bhmdot', 'f8')])

# extra properties of a subhalo / group
extradtype = numpy.dtype([
    ('type', 'u1'),   # 0 for subhalo, 1 for contamination
    ('bhmassive', bhdtype),  # properties of the most massive blackhole 
    ('bhluminous', bhdtype), # properties of the most luminosous blackhole
    ('sfr', 'f4'),           # total star formation rate
    ('bhmass', 'f4'),        # total blackhole mass
    ('bhmdot', 'f4')])       # total blackhole accretion rate

class SnapDir(object):
    def __init__(self, snapid, ROOT):
        """ the default schema is a fake one, generated from header.txt
            subclass of the writeable version of SnapDir use a real schema.

            this is to avoid pulling in gaepsi in this public code
        """
        snapid = int(snapid)
        self.snapid = snapid
        self.subhalodir = ROOT + '/subhalos/%03d' % snapid
        self.subhalofile = self.subhalodir + '/subhalotab.raw'
        self.groupfile = self.subhalodir + '/grouphalotab.raw'
        self.headerfile = self.subhalodir + '/header.txt'
        try:
            for line in file(self.headerfile, 'r'):
                if line.startswith('flag_double(header) = 0'):
                   flag_double = False
                if line.startswith('flag_double(header) = 1'):
                   flag_double = True
                if line.startswith('redshift(header) = '):
                    self.redshift = float(line[19:])
                if line.startswith('boxsize(header) = '):
                    self.boxsize = float(line[18:])
            class fakeschema:
                def __getitem__(self, index, flag_double=flag_double):
                    f = lambda : None
                    if flag_double: f.dtype = numpy.dtype('f8')
                    else: f.dtype = numpy.dtype('f4')
                    return f
            self.schema = fakeschema()
        except IOError:
            self.schema = None

    def readsubhalo(self, g=None):
        """ read the basic subhalo catelog,
            if g is given the catelog is categorized by the group catelog
            g """
        rt = numpy.memmap(self.subhalofile, mode='r', dtype=subdtype)
        if g is not None:
            rt = packarray(rt, g['nhalo'] + 1)
        return rt

    def loadstarLband(self, frame, filter, g=None):
        """ 
            read L_band of all star particles, this will be in memory.

            frame can be 'RfFilter' or 'ObsFilter'.
            filter shall be in format 'SDSS.SDSS/r'

        """
        starmass = self.load(4, 'mass')
        starind = self.load(4, 'SEDindex')
        table = numpy.fromfile(self.filename('4/' + frame, filter),
                dtype='f4')
        rt = table[starind] * starmass 
        if g is None:
            return rt
        rt = packarray(rt, g['lenbytype'][:, 4])
        return rt
        
    def readSEDspectra(self, type, subhaloid):
        """ read the SED of given type (stellar or full) for subhalo id ,
            returns rest frame wavelength (Angstrom?) and the SED.
            return rfwl, SED
            use self.redshfit to convert to obs wave length

            if the halo if given id has no SED, return None

            this is slow.
        """
        f = file(self.subhalodir + '/SED/index.raw', 'r')
        f.seek(subhaloid * 8)
        offset = numpy.fromfile(f, dtype='i8', count=1)
        if offset == -1:
            return None
        else:
            assert len(SEDLAM) == 1220
            f = file(self.subhalodir + '/SED/%s.raw' % type, 'r')
            f.seek(offset * 1220 * 4)
            return SEDLAM, numpy.fromfile(f, count=1220, dtype='f4')

    def readgroup(self):
        """ read the basic group catelog.
            the returned catelog can be used to categorize readsubhalo
            or load.
        """
        return numpy.memmap(self.groupfile, mode='r', dtype=groupdtype)

    def load(self, type, comp, g=None):
        """ this will read in property 'comp' of partile type type.

            See pdtype for the basic properties of type=0~5.

            Other supported 'comp' for type=0~5 are:
              5: bhmass bhmdot
              4: met sft
              0: sfr

            See extradtype for a list of extra properties supported
            with comp parameter for type='subhalo'.

            If type is 'SED', we read in the band luminosity of subtab
            entries of comp starts with either 'o' (observed frame )
            or 'r' (rest frame), then followed by a band filter set
            (contamination entries have 0 luminosity)

            if g is given (either from readsubhalo or readgroup)
            the returned array A will be chunked so that
            A[0] is the property of particles in the first group
            A[1] is the property of particles in the second group
            so on.
            
            Example 1: read velocity and mass of star particles, 
            and organize them by subhalos, then evaluate
            the total kinetic energy of all stars 
            (AWARE: we using a a wrong formula here)

            snapdir = Snapdir(18, './')
            g = snapdir.readsubhalo()
            v = snapdir.load(4, 'vel', g)
            m = snapdir.load(4, 'mass', g)
           
            mv2 = m * v ** 2
            mv2 = array([i.sum() for i in mv2])
           
            Example 2: read position of blackholes,
            and calculate the auto-correlation

            snapdir = Snapdir(18, './')
            pos = snapdir.load(5, 'pos')
            # need kdcount
            from kdcount import correlate
            data = correlate.points(pos, boxsize=snapdir.boxsize)
            DD, bins = correlate.paircount(data, data, 
                correlate.RBins(20000, 20))
            rcenter = bins.centers
            V = snapdir.boxsize ** 3
            dV = numpy.diff(4 / 3.0 * numpy.pi * bins.edges ** 3)
            RR = (dV / V) 
            DD = DD[:-1]
            corr = DD / RR - 1
            print zip(rcenter, corr)
            
            Example 3: find the most luminous blackhole of
            all subhalos, and print the mass. 
            (This is already calculated and can be loaded 
            with snapdir.load('subhalo', 'bhmassive')

            snapdir = Snapdir(18, './')
            tab = snapdir.readsubhalo()
            bhmass = snapdir.load(5, 'bhmass', g)
            bhmdot = snapdir.load(5, 'bhmdot', g)
            
            Nbh = g['lenbytype'][:, 5]
            iscontamination = numpy.isnan(g['mass'])
            for i in ((Nbh > 0) & ~iscontamination).nonzero()[0]:
                a = bhmdot[i].argmax()
                print 'subhalo', i, 'most massive bhmass', bhmass[i][a]
            
            to readin extra properties of subhalos, use
            snapdir.load('subhalo', 'bhmassive')
            (for the most massive bh in the subhalo)
            or
            snapdir.load('subhalo', 'bhluminous')
            (for the most luminous bh in the subhalo)
            snapdir.load('subhalo', 'sfr')
            (for the sfr of the subhalo)

            Example 4: read the predicted SDSS luminosity
                and histogram on the rest-frame 'r' band, 
                filtering out those dimmer than 1.0. 

            snapdir = Snapdir(18, './')
            r = snapdir.load('subhalo', 'RfFilter/SDSS.SDSS/r')
            # note that contaminations have 0 luminosity, skipped
            # automatically.
            mask = r > 1.0
            print numpy.histogram(r[mask])
        """
        if not isinstance(type, basestring):
            itype = type
            type = '%d' % type

        if type == 'group':
            if type == 'tab':
                return self.readgroup()
            itype = None
        elif type == 'subhalo':
            if comp == 'tab':
                return self.readsubhalo(g=g)
            if comp in extradtype.names:
                dtype = extradtype[comp]
            elif comp.startswith('RfFilter') or comp.startswith('ObsFilter'):
                dtype = numpy.dtype('f4')
            else:
                raise KeyError('component `%s` for subhalo property is unknown' % comp)
            itype = None
        elif type in '012345':
            if comp in pdtype.names:
                dtype = pdtype[comp]
            else:
                print comp,pdtype.names
                dtype=self.schema[comp].dtype
            itype = int(type)
        else:
            raise KeyError('type has to be "subhalo" or 0 - 5')

        size = os.path.getsize(self.filename(type, comp))
        if size == 0:
            rt = numpy.fromfile(self.filename(type, comp),
                    dtype=dtype)
        else:
            rt = numpy.memmap(self.filename(type, comp),
                            mode='r',
                            dtype=dtype)
        if g is None:
            return rt
        if itype is not None:
            rt = packarray(rt, g['lenbytype'][:, itype])
        else:
            rt = packarray(rt, g['nhalo'] + 1)
        return rt
    def open(self, type, comp, mode='r'):
        return file(self.filename(type, comp), mode=mode)
    def filename(self, type, comp):
        """ the file name of a type/comp """
        if isinstance(type, basestring):
            return self.subhalodir + '/%s/%s.raw' % (type, comp)
        else:
            return self.subhalodir + '/%d/%s.raw' % (type, comp)

class packarray(numpy.ndarray):
  """ A packarray packs/copies several arrays into the same memory chunk.

      It feels like a list of arrays, but because the memory chunk is continuous,
      
      arithmatic operations are easier to use(via packarray)
  """
  def __new__(cls, array, start=None, end=None):
    """ if end is none, start contains the sizes. 
        if start is also none, array is a list of arrays to concatenate
    """
    self = array.view(type=cls)
    if end is None and start is None:
      start = numpy.array([len(arr) for arr in array], dtype='intp')
      array = numpy.concatenate(array)
    if end is None:
      sizes = start
      self.start = numpy.zeros(shape=len(sizes), dtype='intp')
      self.end = numpy.zeros(shape=len(sizes), dtype='intp')
      self.end[:] = sizes.cumsum()
      self.start[1:] = self.end[:-1]
    else:
      self.start = start
      self.end = end
    self.A = array
    return self
  @classmethod
  def adapt(cls, source, template):
    """ adapt source to a packarray according to the layout of template """
    if not isinstance(template, packarray):
      raise TypeError('template must be a packarray')
    return cls(source, template.start, template.end)

  def __repr__(self):
    return 'packarray: %s, start=%s, end=%s' % \
          (repr(self.A), 
           repr(self.start), repr(self.end))
  def __str__(self):
    return repr(self)

  def copy(self):
    return packarray(self.A.copy(), self.start, self.end)

  def compress(self, mask):
    count = self.end - self.start
    realmask = numpy.repeat(mask, count)
    return packarray(self.A[realmask], self.start[mask], self.end[mask])

  def __getitem__(self, index):
    if isinstance(index, basestring):
      return packarray(self.A[index], self.end - self.start)

    if isinstance(index, slice) :
      start, end, step = index.indices(len(self))
      if step == 1:
        return packarray(self.A[self.start[start]:self.end[end]],
            self.start[start:end] - self.start[start],
            self.end[start:end] - self.start[start])

    if isinstance(index, (list, numpy.ndarray)):
      return packarray(self.A, self.start[index], self.end[index])

    if numpy.isscalar(index):
      start, end = self.start[index], self.end[index]
      if end > start: return self.A[start:end]
      else: return numpy.empty(0, dtype=self.A.dtype)
    raise IndexError('unsupported index type %s' % type(index))

  def __len__(self):
    return len(self.start)

  def __iter__(self):
    for i in range(len(self.start)):
      yield self[i]

  def __reduce__(self):
    return packarray, (self.A, self.end - self.start)

  def __array_wrap__(self, outarr, context=None):
    return packarray.adapt(outarr.view(numpy.ndarray), self)

def main():
    """
        Example

        1. Dump subhalo stellar mass and sdss k band luminosity

            python readsubhalo.py 014 subhalo tab/massbytype/4 RfFilter/SDSS.SDSS/i

        2. Dump subhalo stellar mass and sdss k band luminosity

            python readsubhalo.py 014 subhalo tab/massbytype/4 RfFilter/SDSS.SDSS/i

    """
    import argparse
    from sys import stdout
    from sys import argv
    if len(argv) == 1: return
    from itertools import izip
    ap = argparse.ArgumentParser(" - dump properties ", epilog=main.__doc__)
    ap.add_argument("--prefix", default="../")
    ap.add_argument("snapid")
    ap.add_argument(dest="ptype", choices=[
        'group', 
        'subhalo', 
        '0', 
        '1', 
        '2', 
        '3', 
        '4', 
        '5',
        ])
    ap.add_argument("fields", nargs='+', metavar='path/to/variable')

    ap.add_argument("--include-contamination", 
                action="store_true", dest="no_skip_contamination", default=False)

    A = ap.parse_args()
    print A.fields, A.ptype
    snap = SnapDir(A.snapid, A.prefix)
    comps = ['/'.join(f.split('/')[:-1]) for f in A.fields]
    names = [f.split('/')[-1] for f in A.fields]
    data = []
    fmt = []
    for field in A.fields:
        words = field.split('/')
        for split in range(len(words), 0, -1):
            try:
                comp = '/'.join(words[:split])
                d = snap.load(A.ptype, comp)
                for i in range(split, len(words)):
                    try:
                        index = int(words[i])
                        d = d[:, index]
                    except:
                        d = d[words[i]]
                break
            except:
                continue
        d = d.view(dtype=flatten_dtype(d.dtype))
        data.append(d)
        fmt.append(mkfmtstr(d.dtype, {}, ' '))
    type = snap.load('subhalo', 'type')
    N = len(type)

    stdout.write('#')
    for field in A.fields:
        stdout.write(field)
        stdout.write(' ')
    stdout.write('\n')

    skip_contamination = not (A.no_skip_contamination or A.ptype != 'subhalo')
    if skip_contamination:
        for entry in izip(type, *data):
            if skip_contamination and entry[0] == 1: continue
            for f, d in izip(fmt, entry[1:]):
                stdout.write(f % tuple(d))
                stdout.write(' ')
            stdout.write('\n')
    else:
        for entry in izip(*data):
            for f, d in izip(fmt, entry):
                stdout.write(f % tuple(d))
                stdout.write(' ')
            stdout.write('\n')

def simplerepr(i):
    if len(i) == 0:
        return ''
    if len(i) == 1:
        return '(' + str(i[0]) + ')'
    return '(' + str(i) + ')'

def flatten_dtype(dtype, _next=None):
    """ Unpack a structured data-type.  """
    types = []
    if _next is None: 
        _next = [0, '']
        primary = True
    else:
        primary = False

    prefix = _next[1]

    if dtype.names is None:
        for i in numpy.ndindex(dtype.shape):
            if dtype.base == dtype:
                types.append(('%s%s' % (prefix, simplerepr(i)), dtype))
                _next[0] += 1
            else:
                _next[1] = '%s%s' % (prefix, simplerepr(i))
                types.extend(flatten_dtype(dtype.base, _next))
    else:
        for field in dtype.names:
            typ_fields = dtype.fields[field]
            if len(prefix) > 0:
                _next[1] = prefix + '.' + field
            else:
                _next[1] = '' + field
            flat_dt = flatten_dtype(typ_fields[0], _next)
            types.extend(flat_dt)

    _next[1] = prefix
    if primary:
        return numpy.dtype(types)
    else:
        return types

def mkfmtstr(dtype, prefixfmt, delimiter, defaultfmt={
        'f': '%g'        ,
        'd': '%g'        ,
        'i': '%d'        ,
        'B': '%d'        ,
        'b': '%d'        ,
        'I': '%d'        ,
        'L': '%d'        ,
        'S': '"%s"'        ,}
        ):
    l = []
    for name in dtype.names:
        val = None
        for key in prefixfmt:
            if name.startswith(key):
                val = prefixfmt[key]
                break
        if val is None:
            val = defaultfmt[dtype[name].char]
        l.append(val)
    return delimiter.join(l)

# RF SED wavelength from Stephen's code. 
# duplicated here to avoid depending on lam.npy.
#
SEDLAM = numpy.array([
91, 94, 96, 98, 100, 102, 104, 106, 108, 110,
114, 118, 121, 125, 127, 128, 131, 132, 134, 137,
140, 143, 147, 151, 155, 159, 162, 166, 170, 173,
177, 180, 182, 186, 191, 194, 198, 202, 205, 210,
216, 220, 223, 227, 230, 234, 240, 246, 252, 257,
260, 264, 269, 274, 279, 284, 290, 296, 301, 308,
318, 328, 338, 348, 357, 366, 375, 385, 395, 405,
414, 422, 430, 441, 451, 460, 470, 480, 490, 500,
506, 512, 520, 530, 540, 550, 560, 570, 580, 590,
600, 610, 620, 630, 640, 650, 658, 665, 675, 685,
695, 705, 716, 726, 735, 745, 755, 765, 775, 785,
795, 805, 815, 825, 835, 845, 855, 865, 875, 885,
895, 905, 915, 925, 935, 945, 955, 965, 975, 985,
995, 1005, 1015, 1025, 1035, 1045, 1055, 1065, 1075, 1085,
1095, 1105, 1115, 1125, 1135, 1145, 1155, 1165, 1175, 1185,
1195, 1205, 1215, 1225, 1235, 1245, 1255, 1265, 1275, 1285,
1295, 1305, 1315, 1325, 1335, 1345, 1355, 1365, 1375, 1385,
1395, 1405, 1415, 1425, 1435, 1442, 1447, 1455, 1465, 1475,
1485, 1495, 1505, 1512, 1517, 1525, 1535, 1545, 1555, 1565,
1575, 1585, 1595, 1605, 1615, 1625, 1635, 1645, 1655, 1665,
1672, 1677, 1685, 1695, 1705, 1715, 1725, 1735, 1745, 1755,
1765, 1775, 1785, 1795, 1805, 1815, 1825, 1835, 1845, 1855,
1865, 1875, 1885, 1895, 1905, 1915, 1925, 1935, 1945, 1955,
1967, 1976, 1984, 1995, 2005, 2015, 2025, 2035, 2045, 2055,
2065, 2074, 2078, 2085, 2095, 2105, 2115, 2125, 2135, 2145,
2155, 2165, 2175, 2185, 2195, 2205, 2215, 2225, 2235, 2245,
2255, 2265, 2275, 2285, 2295, 2305, 2315, 2325, 2335, 2345,
2355, 2365, 2375, 2385, 2395, 2405, 2415, 2425, 2435, 2445,
2455, 2465, 2475, 2485, 2495, 2505, 2513, 2518, 2525, 2535,
2545, 2555, 2565, 2575, 2585, 2595, 2605, 2615, 2625, 2635,
2645, 2655, 2665, 2675, 2685, 2695, 2705, 2715, 2725, 2735,
2745, 2755, 2765, 2775, 2785, 2795, 2805, 2815, 2825, 2835,
2845, 2855, 2865, 2875, 2885, 2895, 2910, 2930, 2950, 2970,
2990, 3010, 3030, 3050, 3070, 3090, 3110, 3130, 3150, 3170,
3190, 3210, 3230, 3250, 3270, 3290, 3310, 3330, 3350, 3370,
3390, 3410, 3430, 3450, 3470, 3490, 3510, 3530, 3550, 3570,
3590, 3610, 3630, 3640, 3650, 3670, 3690, 3710, 3730, 3750,
3770, 3790, 3810, 3830, 3850, 3870, 3890, 3910, 3930, 3950,
3970, 3990, 4010, 4030, 4050, 4070, 4090, 4110, 4130, 4150,
4170, 4190, 4210, 4230, 4250, 4270, 4290, 4310, 4330, 4350,
4370, 4390, 4410, 4430, 4450, 4470, 4490, 4510, 4530, 4550,
4570, 4590, 4610, 4630, 4650, 4670, 4690, 4710, 4730, 4750,
4770, 4790, 4810, 4830, 4850, 4870, 4890, 4910, 4930, 4950,
4970, 4990, 5010, 5030, 5050, 5070, 5090, 5110, 5130, 5150,
5170, 5190, 5210, 5230, 5250, 5270, 5290, 5310, 5330, 5350,
5370, 5390, 5410, 5430, 5450, 5470, 5490, 5510, 5530, 5550,
5570, 5590, 5610, 5630, 5650, 5670, 5690, 5710, 5730, 5750,
5770, 5790, 5810, 5830, 5850, 5870, 5890, 5910, 5930, 5950,
5970, 5990, 6010, 6030, 6050, 6070, 6090, 6110, 6130, 6150,
6170, 6190, 6210, 6230, 6250, 6270, 6290, 6310, 6330, 6350,
6370, 6390, 6410, 6430, 6450, 6470, 6490, 6510, 6530, 6550,
6570, 6590, 6610, 6630, 6650, 6670, 6690, 6710, 6730, 6750,
6770, 6790, 6810, 6830, 6850, 6870, 6890, 6910, 6930, 6950,
6970, 6990, 7010, 7030, 7050, 7070, 7090, 7110, 7130, 7150,
7170, 7190, 7210, 7230, 7250, 7270, 7290, 7310, 7330, 7350,
7370, 7390, 7410, 7430, 7450, 7470, 7490, 7510, 7530, 7550,
7570, 7590, 7610, 7630, 7650, 7670, 7690, 7710, 7730, 7750,
7770, 7790, 7810, 7830, 7850, 7870, 7890, 7910, 7930, 7950,
7970, 7990, 8010, 8030, 8050, 8070, 8090, 8110, 8130, 8150,
8170, 8190, 8210, 8230, 8250, 8270, 8290, 8310, 8330, 8350,
8370, 8390, 8410, 8430, 8450, 8470, 8490, 8510, 8530, 8550,
8570, 8590, 8610, 8630, 8650, 8670, 8690, 8710, 8730, 8750,
8770, 8790, 8810, 8830, 8850, 8870, 8890, 8910, 8930, 8950,
8970, 8990, 9010, 9030, 9050, 9070, 9090, 9110, 9130, 9150,
9170, 9190, 9210, 9230, 9250, 9270, 9290, 9310, 9330, 9350,
9370, 9390, 9410, 9430, 9450, 9470, 9490, 9510, 9530, 9550,
9570, 9590, 9610, 9630, 9650, 9670, 9690, 9710, 9730, 9750,
9770, 9790, 9810, 9830, 9850, 9870, 9890, 9910, 9930, 9950,
9970, 9990, 10025, 10075, 10125, 10175, 10225, 10275, 10325, 10375,
10425, 10475, 10525, 10575, 10625, 10675, 10725, 10775, 10825, 10875,
10925, 10975, 11025, 11075, 11125, 11175, 11225, 11275, 11325, 11375,
11425, 11475, 11525, 11575, 11625, 11675, 11725, 11775, 11825, 11875,
11925, 11975, 12025, 12075, 12125, 12175, 12225, 12275, 12325, 12375,
12425, 12475, 12525, 12575, 12625, 12675, 12725, 12775, 12825, 12875,
12925, 12975, 13025, 13075, 13125, 13175, 13225, 13275, 13325, 13375,
13425, 13475, 13525, 13575, 13625, 13675, 13725, 13775, 13825, 13875,
13925, 13975, 14025, 14075, 14125, 14175, 14225, 14275, 14325, 14375,
14425, 14475, 14525, 14570, 14620, 14675, 14725, 14775, 14825, 14875,
14925, 14975, 15025, 15075, 15125, 15175, 15225, 15275, 15325, 15375,
15425, 15475, 15525, 15575, 15625, 15675, 15725, 15775, 15825, 15875,
15925, 15975, 16050, 16150, 16250, 16350, 16450, 16550, 16650, 16750,
16850, 16950, 17050, 17150, 17250, 17350, 17450, 17550, 17650, 17750,
17850, 17950, 18050, 18150, 18250, 18350, 18450, 18550, 18650, 18750,
18850, 18950, 19050, 19150, 19250, 19350, 19450, 19550, 19650, 19750,
19850, 19950, 20050, 20150, 20250, 20350, 20450, 20550, 20650, 20750,
20850, 20950, 21050, 21150, 21250, 21350, 21450, 21550, 21650, 21750,
21850, 21950, 22050, 22150, 22250, 22350, 22450, 22550, 22650, 22750,
22850, 22950, 23050, 23150, 23250, 23350, 23450, 23550, 23650, 23750,
23850, 23950, 24050, 24150, 24250, 24350, 24450, 24550, 24650, 24750,
24850, 24950, 25050, 25150, 25250, 25350, 25450, 25550, 25650, 25750,
25850, 25950, 26050, 26150, 26250, 26350, 26450, 26550, 26650, 26750,
26850, 26950, 27050, 27150, 27250, 27350, 27450, 27550, 27650, 27750,
27850, 27950, 28050, 28150, 28250, 28350, 28450, 28550, 28650, 28750,
28850, 28950, 29050, 29150, 29250, 29350, 29450, 29550, 29650, 29750,
29850, 29950, 30050, 30150, 30250, 30350, 30450, 30550, 30650, 30750,
30850, 30950, 31050, 31150, 31250, 31350, 31450, 31550, 31650, 31750,
31850, 31950, 32100, 32300, 32500, 32700, 32900, 33100, 33300, 33500,
33700, 33900, 34100, 34300, 34500, 34700, 34900, 35100, 35300, 35500,
35700, 35900, 36100, 36300, 36500, 36700, 36900, 37100, 37300, 37500,
37700, 37900, 38100, 38300, 38500, 38700, 38900, 39100, 39300, 39500,
39700, 39900, 40100, 40300, 40500, 40700, 40900, 41100, 41300, 41500,
41700, 41900, 42100, 42300, 42500, 42700, 42900, 43100, 43300, 43500,
43700, 43900, 44100, 44300, 44500, 44700, 44900, 45100, 45300, 45500,
45700, 45900, 46100, 46300, 46500, 46700, 46900, 47100, 47300, 47500,
47700, 47900, 48100, 48300, 48500, 48700, 48900, 49100, 49300, 49500,
49700, 49900, 50100, 50300, 50500, 50700, 50900, 51100, 51300, 51500,
51700, 51900, 52100, 52300, 52500, 52700, 52900, 53100, 53300, 53500,
53700, 53900, 54100, 54300, 54500, 54700, 54900, 55100, 55300, 55500,
55700, 55900, 56100, 56300, 56500, 56700, 56900, 57100, 57300, 57500,
57700, 57900, 58100, 58300, 58500, 58700, 58900, 59100, 59300, 59500,
59700, 59900, 60100, 60300, 60500, 60700, 60900, 61100, 61300, 61500,
61700, 61900, 62100, 62300, 62500, 62700, 62900, 63100, 63300, 63500,
63700, 63900, 64200, 64600, 65000, 65400, 65800, 66200, 66600, 67000,
67400, 67800, 68200, 68600, 69000, 69400, 69800, 70200, 70600, 71000,
71400, 71800, 72200, 72600, 73000, 73400, 73800, 74200, 74600, 75000,
75400, 75800, 76200, 76600, 77000, 77400, 77800, 78200, 78600, 79000,
79400, 79800, 80200, 80600, 81000, 81400, 81800, 82200, 82600, 83000,
83400, 83800, 84200, 84600, 85000, 85400, 85800, 86200, 86600, 87000,
87400, 87800, 88200, 88600, 89000, 89400, 89800, 90200, 90600, 91000,
91400, 91800, 92200, 92600, 93000, 93400, 93800, 94200, 94600, 95000,
95400, 95800, 96200, 96600, 97000, 97400, 97800, 98200, 98600, 99000,
99400, 99800, 100200, 200000, 400000, 600000, 800000, 1e+06, 1.2e+06, 1.4e+06,
    ])

if __name__ == '__main__':
    main()

