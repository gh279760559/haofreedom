# denseCRF 3D segmentation

The main goal is to segment the 3D objects simultaneously when scanning the mesh. With the human interaction of one click on the mesh, the click area will spread out until it reaches to the edge of the object.

## Getting Started


### Prerequisites

tbc...

```
Give examples
```

### Installing

use `git clone` command to get this on your local machine.

## Running the tests

follow the following steps:

1. merge 2 ply files.

    ```
    run merge_plys.py -m fullpath/to/ply_file_label ply_file_color -o fullpath/to/out_file
    ```
   optional: -t if turn on test mode, do `-t 1` default 0.
	     -b if using bilateral pairwise function, default 1.
2. use CloudCompare to manually do 'cut' or 'density reduce'.
3. use CloudCompare to manually select the point on the object.
4. put the index numbers in save_npz_and_visualize_click_area.py (in code, easy to find)
5. run save_npz_and_visualize_click_area.py to save the mesh with the black area in the click point (also include the whole object) into ply file

      ```
     run save_npz_and_visualize_click_area.py -i fullpath/to/ply_file -o temp -p True
     ```
6. run evaluation_framework.py to start learning parameters.

     ```
     run evaluation_framework.py -m fullpath/to/ply_file_label ply_file_color -o out_file
     ```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* tbc...

## Contributing


## Versioning

## Authors

* **Hao Guan**


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
