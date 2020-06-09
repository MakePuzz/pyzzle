! program main
!     implicit none
!     integer, parameter :: width=5, height=5
!     integer :: puzzle(width, height)
!     integer :: enable(width, height)
!     integer, parameter :: n = 6
!     integer ::words(n)
!     integer :: is(n), js(n), oris(n)
!     integer :: i, j, w_len_max
!     integer :: used_index(n)
   
!     puzzle(1, 1:5) = (/"H", "O", "G", "E", blank/)
!     ! puzzle(1, 1:5) = (/blank, blank, blank, blank, blank/)
!     puzzle(2, 1:5) = (/blank, blank, blank, blank, blank/)
!     puzzle(3, 1:5) = (/blank, blank, blank, blank, blank/)
!     puzzle(4, 1:5) = (/blank, blank, blank, blank, blank/)
!     puzzle(5, 1:5) = (/blank, blank, blank, blank, blank/)

!     ! This means a puzzle below:
!     ! H * * * *
!     ! O * * * *
!     ! G * * * *
!     ! E * * * *
!     ! * * * * *
!     enable(1, 1:5) = (/1, 1, 1, 1, 0/)
!     enable(2, 1:5) = (/1, 1, 1, 1, 1/)
!     enable(3, 1:5) = (/1, 1, 1, 1, 1/)
!     enable(4, 1:5) = (/1, 1, 1, 1, 1/)
!     enable(5, 1:5) = (/1, 1, 1, 1, 1/)

!     oris(1)=0
!     js(1)=4
!     is(1)=1
!     words(1)='EEE'

!     ! This means a word below:
!     ! * * * E *
!     ! * * * E *
!     ! * * * E *
!     ! * * * * *
!     ! * * * * *

!     oris(2)=1
!     js(2)=1
!     is(2)=3
!     words(2)='GAME'

!     ! This means a word below:
!     ! * * * * *
!     ! * * * * *
!     ! G A M E *
!     ! * * * * *
!     ! * * * * *

!     oris(3)=0
!     js(3)=3
!     is(3)=3
!     words(3)='MAD'

!     ! This means a word below:
!     ! * * * * *
!     ! * * * * *
!     ! * * M * *
!     ! * * A * *
!     ! * * D * *

!     oris(4)=0
!     js(4)=4
!     is(4)=1
!     words(4)='AAEA'

!     ! This means a word below:
!     ! * * * A *
!     ! * * * A *
!     ! * * * E *
!     ! * * * A *
!     ! * * * * *

!     oris(5)=1
!     js(5)=1
!     is(5)=1
!     words(5)='HAGE'

!     ! This means a word below:
!     ! H A G E *
!     ! * * * * *
!     ! * * * * *
!     ! * * * * *
!     ! * * * * *

!     oris(6)=1
!     js(6)=4
!     is(6)=2
!     words(6)='EQ'

!     ! This means a word below:
!     ! * * * * *
!     ! * * * E Q
!     ! * * * * *
!     ! * * * * *
!     ! * * * * *

!     w_len_max = 4

!     call add_to_limit(puzzle, enable, 5, 5, n, oris, is, js, words, w_len_max, used_index)

!     ! write out
!     do i = 1, height
!         write(*,*) puzzle(1:width, i)
!     end do
!     do i = 1, height
!         write(*,*) enable(1:width, i)
!     end do


! end program main


subroutine add_to_limit(height, width, n, w_len_max, blank, oris, is, js, ks, words, w_lens, puzzle, enable, used_index)
    implicit none
    integer, intent(in) :: height, width
    integer, intent(in) :: n, w_len_max, blank
    integer, dimension(:), intent(in) :: oris, is, js, ks
    integer, dimension(:,:), intent(in) :: words
    integer, dimension(:), intent(in) :: w_lens
    integer, dimension(:,:), intent(inout) :: puzzle
    integer, dimension(:,:), intent(inout) :: enable
    integer, dimension(n), intent(out) :: used_index

    integer :: w_len, a, b, i, j, k, ori, placeability
    integer :: is_placeable
    integer :: word(w_len_max)
    logical :: placed

    logical :: is_already_used
    integer :: c, place_count!
    place_count = 0
    used_index = -1


    ! do i = 1, height
    !     write(*,*) puzzle(1:width, i)
    ! end do

    do while (.true.)
        placed = .false.
        do a = 1, n
            w_len = w_lens(a)
            word = words(a, 1:w_len)
            i = is(a)
            j = js(a)
            k = ks(a)
            ori = oris(a)

            is_already_used = .false.
            if (place_count >= 1) then
                do c = 1, place_count
                    if (k == ks(used_index(c))) then
                        is_already_used = .true.
                        exit
                    end if
                end do
            end if
            if (is_already_used .eqv. .true.) then
                cycle
            end if
            if (used_index(a) == 0) then
                cycle
            end if

            placeability = is_placeable(puzzle, enable, height, width, ori, i, j, word, w_len, blank)    
            ! write(6,*) a, placeability
            if (placeability /= 0) then
                cycle
            end if
            placed = .true.
            place_count = place_count + 1
            used_index(place_count) = a
            ! used_k(place_count) = k
            if (oris(a) == 0) then
                do b = 1, w_len
                    puzzle(is(a)+b-1, js(a)) = word(b) ! place character
                end do
                ! update enable
                if (is(a)-1 >= 1) then
                    enable(is(a)-1, js(a)) = 0
                end if
                if (is(a)+w_len <= width) then
                    enable(is(a)+w_len, js(a)) = 0
                end if
            end if
            if (oris(a) == 1) then
                do b = 1, w_len
                    puzzle(is(a), js(a)+b-1) = word(b) ! place character
                end do
                ! update enable
                if (js(a)-1 >= 1) then
                    enable(is(a), js(a)-1) = 0
                end if
                if (js(a)+w_len <= height) then
                    enable(is(a), js(a)+w_len) = 0
                end if
            end if
        end do

        if (placed .eqv. .false.) then
            exit
        end if
    end do
end subroutine


integer function is_placeable(puzzle, enable, height, width, ori, i, j, word, w_len, blank)
    
    integer, intent(in) :: ori, i, j, w_len, width, height
    integer, intent(in) :: word(w_len)
    integer, intent(in) :: puzzle(height, width)
    integer, intent(in) :: enable(height, width)
    integer, intent(in) :: blank
    
    integer :: check, cf, ce
    integer :: a, b
    logical :: is_cross_at_a, at_least_1_cross
    
    ! ori=0:vertical、ori=1:lateral
    ! i:number of row
    ! j:number of column

    ! nx=ubound(puzzle, dim = 1)
    ! ny=ubound(puzlle, dim=2)
    ! ubound(source, dim = 1)

    !The result number corresponds to the judgment result
    ! 0. The word can be placed (only succeeded)
    ! 1. The preceding and succeeding cells are already filled
    ! 2. At least one place must cross other words
    ! 3. Not a correct intersection
    ! 4. The same word is in use (not used)
    ! 5. The Neighbor cells are filled except at the intersection
    ! 6. US/USA, DOMINICA/DOMINICAN problem
    
    !---1. check if all blanks------------------------------------------------
    check = 0
    do a = 1, width
        do b = 1, height
            if (puzzle(b, a) /= blank) check = 1
        end do
    end do
    if (check == 0) then
        is_placeable = 0 !The word can be placed (only succeeded)
        return
    end if

    !---2. whether there are characters before and after--------------------------------------------
    cf = 0
    ce = 0
    if (ori == 0) then ! vertical
        !First character above
        if (i == 1) then
            cf = 1
        else
            if (puzzle(i-1, j) == blank) then
                cf = 1
            end if
        end if
        !Below the last letter
        if ((i+w_len-1) == height) then
            ce = 1
        else
            if (puzzle(i+w_len, j) == blank) then
                ce = 1
            end if
        end if
    else if (ori == 1) then ! lateral
        !First character left
        if (j == 1) then
            cf = 1
        else
            if (puzzle(i, j-1) == blank) then
                cf = 1
            end if
        end if
        !Last character right
        if ((j+w_len-1) == width) then
            ce = 1
        else
            if (puzzle(i, j+w_len) == blank) then
                ce = 1
            end if
        end if
    end if
    if (cf == 0 .or. ce == 0) then
        is_placeable = 1 !The preceding and succeeding cells are already filled
        return
    end if
    
    !---3.cross other words and crosses match and US/USA problem--------------------
    if (ori == 0) then ! vertical
        at_least_1_cross = .false. !If there is a cross,  at_least_1_cross = .true.
        do a = 1-1, w_len-1
            is_cross_at_a = .false. !If there is a cross,  is_cross_at_a = .true.
            if (puzzle(i+a, j) /= blank) then
                is_cross_at_a = .true.
                at_least_1_cross = .true.
                if (puzzle(i+a, j) == word(a+1)) then
                    cycle !If crosses match, go to next 
                else
                    is_placeable = 3 !Not a correct intersection
                    return
                end if
            else !other than cross
                if (j > 1 .and. puzzle(i+a, j-1) /= blank) then
                    is_placeable = 5 !The Neighbor cells are filled except at the intersection
                    return
                end if
                if (j < width .and. puzzle(i+a, j+1) /= blank) then
                    is_placeable = 5 !The Neighbor cells are filled except at the intersection
                    return
                end if
            end if
        end do
        if (at_least_1_cross .eqv. .false.) then
            is_placeable = 2 !At least one place must cross other words
            return
        end if
        do a = 1-1, w_len-1
            if (enable(i+a, j) == 0) then
                is_placeable = 6 ! US/USA, DOMINICA/DOMINICAN problem
                return
            end if
        end do
    else if (ori == 1) then !lateral
        at_least_1_cross = .false. !If there is a cross,  at_least_1_cross = .true.
        do a = 1-1, w_len-1
            is_cross_at_a = .false.
            if (puzzle(i, j+a) /= blank) then
                is_cross_at_a = .true.
                at_least_1_cross = .true.
                if (puzzle(i, j+a) == word(a+1)) then
                    cycle !If crosses match, go to next 
                else
                    is_placeable = 3 !Not a correct intersection
                    return
                end if
            else !other than cross
                if (i > 1 .and. puzzle(i-1, j+a) /= blank) then
                    is_placeable = 5 !The Neighbor cells are filled except at the intersection
                    return
                end if
                if (i < height .and. puzzle(i+1, j+a) /= blank) then
                    is_placeable = 5 !The Neighbor cells are filled except at the intersection
                    return
                end if
            end if
        end do
        if (at_least_1_cross .eqv. .false.) then
            is_placeable = 2 !At least one place must cross other words
            return
        end if
        do a = 1-1, w_len-1
            if (enable(i, j+a) == 0) then
                is_placeable = 6 !US/USA, DOMINICA/DOMINICAN problem
                return
            end if
        end do
    end if

    is_placeable = 0
    return
end function