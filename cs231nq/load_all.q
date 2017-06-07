/ load all non run/load[data] scripts

toLoad:distinct enlist["nn_util.q"],{x where not any x like/: ("run*";"load*")} system"ls *.q"
{@[{-1"loading ",x;system"l ",x;};x;{"failed to load due to ",x}]}each toLoad;
