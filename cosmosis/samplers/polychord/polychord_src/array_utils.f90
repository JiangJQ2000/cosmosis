module array_module
    use utils_module, only: dp

    interface add_point
        module procedure add_point_3d, add_point_2d
    end interface add_point
    interface delete_point
        module procedure delete_point_3d, delete_point_2d
    end interface delete_point
    interface reallocate
        module procedure reallocate_1_d, reallocate_2_d, reallocate_3_d, reallocate_1_i
    end interface reallocate
    contains

    !> Reallocate a 1D array of doubles
    !!
    !! The array is allocated with a new size defined by new_size1 (optional variable)
    !!
    !! The indices to be transferred across are defined by save_indices1 (optional variable)
    !!
    !! The positions of the data in the reallocated array are defined by target_indices1 (optional variable)
    subroutine reallocate_1_d(array,new_size1,save_indices1,target_indices1)
        use abort_module, only: halt_program
        implicit none
        real(dp), dimension(:),allocatable, intent(inout) :: array !> Array to be reallocated
        integer, intent(in),optional :: new_size1                          !> New size of the array
        integer, dimension(:),intent(in),optional :: save_indices1         !> indices of old array to be transferred
        integer, dimension(:),intent(in),optional :: target_indices1       !> new array indices after reallocation

        ! Temporary versions of the above variables
        real(dp), dimension(size(array,1)) :: a
        integer :: n1
        integer, allocatable, dimension(:) :: s1
        integer, allocatable, dimension(:) :: t1

        ! constructor variable
        integer :: i

        ! Check to see that it is already allocated
        if(.not.allocated(array)) call halt_program('reallocate_1_d error: array is not allocated') 

        ! Define the new size of the array
        if( present(new_size1) ) then
            n1 = new_size1     ! If the argument is present, then this is the reallocation size
        else
            n1 = size(array,1) ! Default reallocation size is the size of the original array
        end if

        ! Define the positions of where data is coming from and going to
        if(present(save_indices1) .and. present(target_indices1)) then
            s1 = save_indices1
            t1 = target_indices1
            if(size(s1)/=size(t1)) call halt_program('reallocate_1_d error: save and target indices must be equal in size') 
        else if(present(save_indices1)) then
            s1 = save_indices1
            t1 = [ (i,i=1,size(s1)) ]
        else if(present(target_indices1)) then
            t1 = target_indices1
            s1 = [ (i,i=1,size(t1)) ]
        else
            s1 = [ (i,i=1,size(array,1)) ]
            t1 = [ (i,i=1,size(array,1)) ]
        end if

        a = array                           ! Save the old array 
        deallocate(array)                   ! Deallocate it      
        allocate(array(n1))                 ! Re-allocate with new size
        array(t1) = a(s1)                   ! Reassign the old values

    end subroutine reallocate_1_d


    !> Reallocate a 2D array of doubles
    !!
    !! The array is allocated with a new size defined by new_size1,new_size2 (optional variables)
    !!
    !! The indices to be transferred across are defined by save_indices1,save_indices2 (optional variables)
    !!
    !! The positions of the data in the reallocated array are defined by target_indices1,target_indices2, (optional variables)
    subroutine reallocate_2_d(array,new_size1,new_size2,save_indices1,save_indices2,target_indices1,target_indices2)
        use abort_module, only: halt_program
        implicit none
        !> Array to be reallocated
        real(dp), dimension(:,:),allocatable, intent(inout) :: array
        !> New size of the array 
        integer, intent(in),optional :: new_size1,new_size2
        !> indices of old array to be transferred
        integer, dimension(:),intent(in),optional :: save_indices1,save_indices2
        !> new array indices after reallocation 
        integer, dimension(:),intent(in),optional :: target_indices1,target_indices2

        ! Temporary versions of the above variables
        real(dp), dimension(size(array,1),size(array,2)) :: a
        integer :: n1,n2
        integer, allocatable, dimension(:) :: s1,s2
        integer, allocatable, dimension(:) :: t1,t2

        ! constructor variable
        integer :: i

        ! Check to see that it is already allocated
        if(.not.allocated(array)) call halt_program('reallocate_2_d error: array is not allocated') 

        ! Define the new size of the array
        if( present(new_size1) ) then
            n1 = new_size1     ! If the argument is present, then this is the reallocation size
        else
            n1 = size(array,1) ! Default reallocation size is the size of the original array
        end if
        if( present(new_size2) ) then
            n2 = new_size2     ! If the argument is present, then this is the reallocation size
        else
            n2 = size(array,2) ! Default reallocation size is the size of the original array
        end if

        ! Define the positions of where data is coming from and going to
        if(present(save_indices1) .and. present(target_indices1)) then
            s1 = save_indices1
            t1 = target_indices1
            if(size(s1)/=size(t1)) call halt_program('reallocate_2_d error: save and target indices must be equal in size') 
        else if(present(save_indices1)) then
            s1 = save_indices1
            t1 = [ (i,i=1,size(s1)) ]
        else if(present(target_indices1)) then
            t1 = target_indices1
            s1 = [ (i,i=1,size(t1)) ]
        else
            s1 = [ (i,i=1,size(array,1)) ] 
            t1 = [ (i,i=1,size(array,1)) ] 
        end if


        ! Define the positions of where data is coming from and going to
        if(present(save_indices2) .and. present(target_indices2)) then
            s2 = save_indices2
            t2 = target_indices2
            if(size(s2)/=size(t2)) call halt_program('reallocate_2_d error: save and target indices must be equal in size') 
        else if(present(save_indices2)) then
            s2 = save_indices2
            t2 = [ (i,i=1,size(s2)) ]
        else if(present(target_indices2)) then
            t2 = target_indices2
            s2 = [ (i,i=1,size(t2)) ]
        else
            s2 = [ (i,i=1,size(array,2)) ] 
            t2 = [ (i,i=1,size(array,2)) ] 
        end if

        a = array                           ! Save the old array 
        deallocate(array)                   ! Deallocate it      
        allocate(array(n1,n2))              ! Re-allocate with new size
        array(t1,t2)= a(s1,s2)              ! Reassign the old values

    end subroutine reallocate_2_d


    !> Reallocate a 3D array of doubles
    !!
    !! The array is allocated with a new size defined by new_size1,new_size2,new_size3 (optional variables)
    !!
    !! The indices to be transferred across are defined by save_indices1,save_indices2,save_indices3 (optional variables)
    !!
    !! The positions of the data in the reallocated array are defined by target_indices1,target_indices2,target_indices3 (optional variables)
    subroutine reallocate_3_d(array,&
                                new_size1,new_size2,new_size3,&
                                save_indices1,save_indices2,save_indices3,&
                                target_indices1,target_indices2,target_indices3)
        use abort_module, only: halt_program
        implicit none
        !> Array to be reallocated
        real(dp), dimension(:,:,:),allocatable, intent(inout) :: array
        !> New size of the array 
        integer, intent(in),optional :: new_size1,new_size2,new_size3
        !> indices of old array to be transferred
        integer, dimension(:),intent(in),optional :: save_indices1,save_indices2,save_indices3
        !> new array indices after reallocation 
        integer, dimension(:),intent(in),optional :: target_indices1,target_indices2,target_indices3

        ! Temporary versions of the above variables
        real(dp), dimension(size(array,1),size(array,2),size(array,3)) :: a
        integer :: n1,n2,n3
        integer, allocatable, dimension(:) :: s1,s2,s3
        integer, allocatable, dimension(:) :: t1,t2,t3

        ! constructor variable
        integer :: i

        ! Check to see that it is already allocated
        if(.not.allocated(array)) call halt_program('reallocate_3_d error: array is not allocated') 

        ! Define the new size of the array
        if( present(new_size1) ) then
            n1 = new_size1     ! If the argument is present, then this is the reallocation size
        else
            n1 = size(array,1) ! Default reallocation size is the size of the original array
        end if
        if( present(new_size2) ) then
            n2 = new_size2     ! If the argument is present, then this is the reallocation size
        else
            n2 = size(array,2) ! Default reallocation size is the size of the original array
        end if
        if( present(new_size3) ) then
            n3 = new_size3     ! If the argument is present, then this is the reallocation size
        else
            n3 = size(array,3) ! Default reallocation size is the size of the original array
        end if

        ! Define the positions of where data is coming from and going to
        if(present(save_indices1) .and. present(target_indices1)) then
            s1 = save_indices1
            t1 = target_indices1
            if(size(s1)/=size(t1)) call halt_program('reallocate_3_d error: save and target indices must be equal in size') 
        else if(present(save_indices1)) then
            s1 = save_indices1
            t1 = [ (i,i=1,size(s1)) ]
        else if(present(target_indices1)) then
            t1 = target_indices1
            s1 = [ (i,i=1,size(t1)) ]
        else
            s1 = [ (i,i=1,size(array,1)) ] 
            t1 = [ (i,i=1,size(array,1)) ] 
        end if


        ! Define the positions of where data is coming from and going to
        if(present(save_indices2) .and. present(target_indices2)) then
            s2 = save_indices2
            t2 = target_indices2
            if(size(s2)/=size(t2)) call halt_program('reallocate_3_d error: save and target indices must be equal in size') 
        else if(present(save_indices2)) then
            s2 = save_indices2
            t2 = [ (i,i=1,size(s2)) ]
        else if(present(target_indices2)) then
            t2 = target_indices2
            s2 = [ (i,i=1,size(t2)) ]
        else
            s2 = [ (i,i=1,size(array,2)) ] 
            t2 = [ (i,i=1,size(array,2)) ] 
        end if


        ! Define the positions of where data is coming from and going to
        if(present(save_indices3) .and. present(target_indices3)) then
            s3 = save_indices3
            t3 = target_indices3
            if(size(s3)/=size(t3)) call halt_program('reallocate_3_d error: save and target indices must be equal in size') 
        else if(present(save_indices3)) then
            s3 = save_indices3
            t3 = [ (i,i=1,size(s3)) ]
        else if(present(target_indices3)) then
            t3 = target_indices3
            s3 = [ (i,i=1,size(t3)) ]

        else
            s3 = [ (i,i=1,size(array,3)) ] 
            t3 = [ (i,i=1,size(array,3)) ] 
        end if


        a = array                           ! Save the old array 
        deallocate(array)                   ! Deallocate it      
        allocate(array(n1,n2,n3))           ! Re-allocate with new size
        array(t1,t2,t3)= a(s1,s2,s3)        ! Reassign the old values

    end subroutine reallocate_3_d




    !> Reallocate a 1D array of integers
    !!
    !! The array is allocated with a new size defined by new_size1 (optional variable)
    !!
    !! The indices to be transferred across are defined by save_indices1 (optional variable)
    !!
    !! The positions of the data in the reallocated array are defined by target_indices1 (optional variable)
    subroutine reallocate_1_i(array,new_size1,save_indices1,target_indices1)
        use abort_module, only: halt_program
        implicit none
        integer,          dimension(:),allocatable, intent(inout) :: array !> Array to be reallocated
        integer, intent(in),optional :: new_size1                          !> New size of the array
        integer, dimension(:),intent(in),optional :: save_indices1         !> indices of old array to be transferred
        integer, dimension(:),intent(in),optional :: target_indices1       !> new array indices after reallocation

        ! Temporary versions of the above variables
        integer, dimension(size(array,1)) :: a
        integer :: n1
        integer, allocatable, dimension(:) :: s1
        integer, allocatable, dimension(:) :: t1

        ! constructor variable
        integer :: i

        ! Check to see that it is already allocated
        if(.not.allocated(array)) call halt_program('reallocate_1_i error: array is not allocated') 

        ! Define the new size of the array
        if( present(new_size1) ) then
            n1 = new_size1     ! If the argument is present, then this is the reallocation size
        else
            n1 = size(array,1) ! Default reallocation size is the size of the original array
        end if

        ! Define the positions of where data is coming from and going to
        if(present(save_indices1) .and. present(target_indices1)) then
            s1 = save_indices1
            t1 = target_indices1
            if(size(s1)/=size(t1)) call halt_program('reallocate_1_i error: save and target indices must be equal in size') 
        else if(present(save_indices1)) then
            s1 = save_indices1
            t1 = [ (i,i=1,size(s1)) ]
        else if(present(target_indices1)) then
            t1 = target_indices1
            s1 = [ (i,i=1,size(t1)) ]
        else
            s1 = [ (i,i=1,size(array,1)) ] 
            t1 = [ (i,i=1,size(array,1)) ] 
        end if

        a = array                           ! Save the old array 
        deallocate(array)                   ! Deallocate it      
        allocate(array(n1))                 ! Re-allocate with new size
        array(t1) = a(s1)                   ! Reassign the old values

    end subroutine reallocate_1_i





















    subroutine add_point_3d(point,array,narray,cluster_id)
        implicit none
        real(dp), dimension(:), intent(in) :: point                     !> Point to be added to end of array
        real(dp), dimension(:,:,:), allocatable, intent(inout) :: array !> Array to be added to
        integer,dimension(:), allocatable, intent(inout) :: narray              !> number of points in array (second index)
        integer, intent(in) :: cluster_id                                       !> cluster identity (third index)
        integer :: newsize2

        narray(cluster_id) = narray(cluster_id) + 1         ! Increase the number of points in the cluster

        ! If this takes us over the size of the array, then double it
        newsize2 = max(1,size(array,2)*2)
        if(narray(cluster_id) > size(array,2) ) call reallocate(array, new_size2=newsize2 )

        array(:,narray(cluster_id),cluster_id) = point       ! Add the point to the end position

    end subroutine add_point_3d

    subroutine add_point_2d(point,array,narray)
        implicit none
        real(dp), dimension(:), intent(in) :: point                   !> Point to be added to end of array
        real(dp), dimension(:,:), allocatable, intent(inout) :: array !> Array to be added to
        integer, intent(inout) :: narray                                      !> number of points in array (second index)
        integer :: newsize2

        narray = narray + 1         ! Increase the number of points

        ! If this takes us over the size of the array, then double it
        newsize2 = max(1,size(array,2)*2)
        if(narray > size(array,2) ) call reallocate(array, new_size2=newsize2 )

        array(:,narray) = point       ! Add the point to the end position

    end subroutine add_point_2d



    function delete_point_3d(i_point,array,narray,cluster_id) result(point)
        implicit none
        integer, intent(in) :: i_point                                          !> Position of point to be deleted from array
        real(dp), dimension(:,:,:), allocatable, intent(inout) :: array !> Array to be delete from
        integer,dimension(:), allocatable, intent(inout) :: narray              !> number of points in array (second index)
        integer, intent(in) :: cluster_id                                       !> cluster identity (third index)
        real(dp), dimension(size(array,1)) :: point                     ! The point we have just deleted

        point = array(:,i_point,cluster_id)                                  ! Output the point to be deleted
        array(:,i_point,cluster_id) = array(:,narray(cluster_id),cluster_id) ! delete the point by overwriting it with the point at the end 
        narray(cluster_id) = narray(cluster_id) - 1                          ! reduce the number of points in the cluster

    end function delete_point_3d

    function delete_point_2d(i_point,array,narray) result(point)
        implicit none
        integer, intent(in) :: i_point                                          !> Position of point to be deleted from array
        real(dp), dimension(:,:), allocatable, intent(inout) :: array   !> Array to be delete from
        integer, intent(inout) :: narray                                        !> number of points in array (second index)
        real(dp), dimension(size(array,1)) :: point                     ! The point we have just deleted

        point = array(:,i_point)           ! Output the point to be deleted
        array(:,i_point) = array(:,narray) ! delete the point by overwriting it with the point at the end 
        narray = narray - 1                ! reduce the number of points in the cluster

    end function delete_point_2d




end module array_module
