# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# don't put duplicate lines in the history. See bash(1) for more options
# ... or force ignoredups and ignorespace
HISTCONTROL=ignoredups:ignorespace

# append to the history file, don't overwrite it
shopt -s histappend checkwinsize

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
# HISTSIZE=1000
# HISTFILESIZE=2000

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

if [ -f /etc/bash_completion ] && ! shopt -oq posix; then
    . /etc/bash_completion
fi

export PS1='[ \[\e[1;33m\]\# \[\e[0;22m\]- \[\e[1;31m\]\u@\h \[\e[0;22m\]- \[\e[1;33m\]\@ \[\e[0;22m\]- \[\e[1;35m\]$(/etc/compute-dir-size.py)\[\e[0;22m\] : \[\e[1;36m\]$(pwd) \[\e[0;22m\]] $ \[\e[m\]\[\e[0;22m\]'
