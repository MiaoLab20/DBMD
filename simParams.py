parmFile = "test/gcaa_unfold.top"
crdFile = "test/gcaa_unfold.crd"
simType = "RNA.implicit"    # "explicit", "protein.implicit", "RNA.implicit"

nbCutoff = 9.0      # if "explicit"
temperature = 300

ntcmd = 15000000
cmdRestartFreq = 500

ncycebprepstart, ncycebprepend = 0, 2
ntebpreppercyc = 5000000
ebprepRestartFreq = 500

ncycebstart, ncycebend = 0, 3
ntebpercyc = 5000000
ebRestartFreq = 500

ncycprodstart, ncycprodend = 0, 5
ntprodpercyc = 100000000
prodRestartFreq = 500

refED_factor, refEP_factor = 0.0, 0.025      # between 0 and 1
