parmFile = "test/gcaa_unfold.top"
crdFile = "test/gcaa_unfold.crd"
simType = "RNA.implicit"    # "explicit", "protein.implicit", "RNA.implicit"

nbCutoff = 9.0      # if "explicit"
temperature = 300

ntcmd = 10000000
cmdRestartFreq = 100

ncycebprepstart, ncycebprepend = 0, 1
ntebpreppercyc = 2500000
ebprepRestartFreq = 100

ncycebstart, ncycebend = 0, 3
ntebpercyc = 2500000
ebRestartFreq = 100

ncycprodstart, ncycprodend = 0, 4
ntprodpercyc = 250000000
prodRestartFreq = 500

refEP_factor, refED_factor = 0.05, 0.05      # between 0 and 1
