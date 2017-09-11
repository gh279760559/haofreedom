    #include <pcl/pcl_base.h>
    #include <pcl/console/parse.h>
    #include <pcl/point_types.h>
    #include <pcl/io/pcd_io.h>
    #include <pcl/visualization/pcl_visualizer.h>
    #include <pcl/segmentation/supervoxel_clustering.h>
    #include <pcl/io/ply_io.h>
    #include <pcl/visualization/cloud_viewer.h>

    // Types
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

    void
    printText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_arg)
    {
        if (!viewer_arg->updateText ("Left click: rotate view \n"
                                             "Right click: zoom in/out \n"
                                             "Middle click: moving around the object \n"
                                             "Press h for more help (will show "
                                             "in the terminal and some may not work)", 5, 84, 12, 1.0, 1.0, 1.0, "hud_text"))
        viewer_arg->addText ("Left click: rotate view \n"
                                     "Right click: zoom in/out \n"
                                     "Middle click: moving around the object \n"
                                     "Press h for more help (will show "
                                     "in the terminal and some may not work)", 5, 84, 12, 1.0, 1.0, 1.0, "hud_text");
    }

    int main(int argc, char** argv)
    {
        /// **********************************
        ///
        ///
        /// Now Loading the point cloud
        ///
        ///
        /// **********************************

        /// Get pcd or ply path from command line
        std::string inputFilename = argv[1];
        std::string ext("");
        ext = inputFilename;
        size_t sep = ext.find_last_of ('.');
        if (sep != std::string::npos)
            ext = ext.substr (sep+1);
        PCL_INFO ("Loading point cloud\n");
        /// check if the provided pcd file contains normals
        PointCloudT::Ptr inputCloudPtr (new pcl::PointCloud<PointT>);
        pcl::PointCloud<pcl::Normal>::Ptr inputNormalsPtr (new pcl::PointCloud<pcl::Normal>);
        pcl::PCLPointCloud2 inputPointCloud2;
        if (ext.compare("pcd") == 0){
            if (pcl::io::loadPCDFile (inputFilename, inputPointCloud2)){
                PCL_ERROR ("ERROR: Could not read input point cloud %s.\n", inputFilename.c_str ());
                return (3);
            }
        }
        if (ext.compare("ply") == 0){
            if (pcl::io::loadPLYFile (inputFilename, inputPointCloud2)){
                PCL_ERROR ("ERROR: Could not read input point cloud %s.\n", inputFilename.c_str ());
                return (3);
            }
        }
        pcl::fromPCLPointCloud2 (inputPointCloud2, *inputCloudPtr);
        PCL_INFO ("Done making cloud\n");
        PCL_INFO("Number of points: %d", inputCloudPtr->size());
        PCL_INFO("\n");
        /// **********************************
        ///
        ///
        /// Visualization
        ///
        ///
        /// **********************************
            /// now visulize
            pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
            viewer->setBackgroundColor(0, 0, 0);
            viewer->addPointCloud<pcl::PointXYZRGBA>(inputCloudPtr, "original ply"); //inputCloudPtr, labeledCloud

            /// Visualization Loop
            PCL_INFO ("Loading viewer\n");
            while (!viewer->wasStopped()) {
                viewer->spinOnce(100);
                /// Show origin or Supervoxels
                printText (viewer);
                boost::this_thread::sleep (boost::posix_time::microseconds (100000));
            }
        return 0;
    }

