find . -name 'machine.file.*' -delete

for f in slurm-*.out
do
  mv $f finallogs/$f
done
