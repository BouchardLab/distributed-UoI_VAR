# Distributed-UoI_VAR


## Requirements

languages: C, C++

API and Libraries: MPI, HDF5-parallel, eigen3, gsl


## Installation

1. Clone the module into your directory
2. source load.sh
3. make debug=0 (for no-debug execution)

## Usage

1. run: srun -n 2 -c 4 --cpu_bind=cores -u ./uoi_var ./simdata4016.csv 7 1 3 5 5 output.h5 lasso.h5

TODO: Generate output files and write results. Caluculate timing and report them!

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

TODO: Write history

## Credits

Author: Mahesh Balasubramanian (guidance from Kris Bouchard, Prabhat, Brandon Cook, Trevor Ruiz)

Version: 2.0


## License

TODO: Write license

## Detailed description of the directory

1. load.sh : Has the required modules for the correct execution of UoI_VAR application

