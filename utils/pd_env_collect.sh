#!/usr/bin/env bash

unset GREP_OPTIONS

set -u  # Check for undefined variables

die() {
  # Print a message and exit with code 1.
  #
  # Usage: die <error_message>
  #   e.g., die "Something bad happened."

  echo $@
  exit 1
}

echo "Collecting system information..."

OUTPUT_FILE=pd_env.txt
python_bin_path=$(which python || which python3 || die "Cannot find Python binary")

{
echo
echo '== check python ==================================================='
} >> ${OUTPUT_FILE}

cat <<EOF > /tmp/check_python.py
import platform
print("""python version: %s
python branch: %s
python build version: %s
python compiler version: %s
python implementation: %s
""" % (
platform.python_version(),
platform.python_branch(),
platform.python_build(),
platform.python_compiler(),
platform.python_implementation(),
))
EOF
${python_bin_path} /tmp/check_python.py 2>&1  >> ${OUTPUT_FILE}

{
echo
echo '== check os platform ==============================================='
} >> ${OUTPUT_FILE}

cat <<EOF > /tmp/check_os.py
import platform
print("""os: %s
os kernel version: %s
os release version: %s
os platform: %s
linux distribution: %s
linux os distribution: %s
mac version: %s
uname: %s
architecture: %s
machine: %s
""" % (
platform.system(),
platform.version(),
platform.release(),
platform.platform(),
platform.linux_distribution(),
platform.dist(),
platform.mac_ver(),
platform.uname(),
platform.architecture(),
platform.machine(),
))
EOF
${python_bin_path} /tmp/check_os.py 2>&1  >> ${OUTPUT_FILE}

{
  echo
  echo '== are we in docker ============================================='
  num=`cat /proc/1/cgroup | grep docker | wc -l`;
  if [ $num -ge 1 ]; then
    echo "Yes"
  else
    echo "No"
  fi

  echo
  echo '== compiler ====================================================='
  c++ --version 2>&1

  echo
  echo '== check pips ==================================================='
  pip list 2>&1 | grep "proto\|numpy\|paddlepaddle"


  echo
  echo '== check for virtualenv ========================================='
  ${python_bin_path} -c "import sys;print(hasattr(sys, \"real_prefix\"))"

  echo
  echo '== paddlepaddle import ============================================'
} >> ${OUTPUT_FILE}

cat <<EOF > /tmp/check_pd.py
import paddle as pd;
pd.set_device('cpu')
print("pd.version.full_version = %s" % pd.version.full_version)
print("pd.version.commit = %s" % pd.version.commit)
print("pd.__version__ = %s" % pd.__version__)
print("Sanity check: %r" % pd.zeros([1,2,3])[:1])
EOF
${python_bin_path} /tmp/check_pd.py 2>&1  >> ${OUTPUT_FILE}

LD_DEBUG=libs ${python_bin_path} -c "import paddle"  2>>${OUTPUT_FILE} > /tmp/loadedlibs

{
  grep libcudnn.so /tmp/loadedlibs
  echo
  echo '== env =========================================================='
  if [ -z ${LD_LIBRARY_PATH+x} ]; then
    echo "LD_LIBRARY_PATH is unset";
  else
    echo LD_LIBRARY_PATH ${LD_LIBRARY_PATH} ;
  fi
  if [ -z ${DYLD_LIBRARY_PATH+x} ]; then
    echo "DYLD_LIBRARY_PATH is unset";
  else
    echo DYLD_LIBRARY_PATH ${DYLD_LIBRARY_PATH} ;
  fi


  echo
  echo '== nvidia-smi ==================================================='
  nvidia-smi 2>&1

  echo
  echo '== cuda libs  ==================================================='
} >> ${OUTPUT_FILE}

find /usr/local -type f -name 'libcudart*'  2>/dev/null | grep cuda |  grep -v "\\.cache" >> ${OUTPUT_FILE}
find /usr/local -type f -name 'libudnn*'  2>/dev/null | grep cuda |  grep -v "\\.cache" >> ${OUTPUT_FILE}

{
  echo
  echo '== paddlepaddle installed from info =================='
  pip show paddlepaddle-gpu

  echo
  echo '== python version  =============================================='
  echo '(major, minor, micro, releaselevel, serial)'
  python -c 'import sys; print(sys.version_info[:])'

  echo
  echo '== bazel version  ==============================================='
  bazel version
  echo '== cmake version  ==============================================='
  cmake --version
} >> ${OUTPUT_FILE}

# Remove any words with google.
mv $OUTPUT_FILE old-$OUTPUT_FILE
grep -v -i google old-${OUTPUT_FILE} > $OUTPUT_FILE

echo "Wrote environment to ${OUTPUT_FILE}. You can review the contents of that file."
echo "and use it to populate the fields in the github issue template."
echo
echo "cat ${OUTPUT_FILE}"
echo
