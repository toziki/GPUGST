# .bashrc

# User specific aliases and functions

alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

#export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/var/lib/snapd/snap/bin:/root/bin
#export PATH=$PATH:/opt/software/openGauss/script/gspylib/pssh/bin
#export PATH=$PATH:/usr/local/bin/gcc
#export SSH_AUTH_SOCK=/root/gaussdb_tmp/gauss_socket_tmp
#export SSH_AGENT_PID=6104
export PATH=/home/ljy/cmake-3.28.3-linux-x86_64/bin:$PATH
export PATH=/opt/rh/devtoolset-10/root/bin:$PATH
#export FI_VERBS_USE_ODP=1
export CC=/opt/rh/devtoolset-10/root/bin/gcc
export CXX=/opt/rh/devtoolset-10/root/bin/g++