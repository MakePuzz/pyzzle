! program main
!     implicit none
!     integer, parameter :: width=5, height=5
!     integer :: puzzle(width, height)
!     integer :: enable(width, height)
!     integer, parameter :: n = 6
!     integer ::words(n)
!     integer :: is(n), js(n), oris(n)
!     integer :: i, j, w_len_max
!     integer :: is_used(n)
   
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

!     call add_to_limit(puzzle, enable, 5, 5, n, oris, is, js, words, w_len_max, is_used)

!     ! write out
!     do i = 1, height
!         write(*,*) puzzle(1:width, i)
!     end do
!     do i = 1, height
!         write(*,*) enable(1:width, i)
!     end do


! end program main


subroutine add_to_limit(height, width, n, w_len_max, blank, oris, is, js, ks, words, w_lens, puzzle, enable, is_used)
    implicit none
    integer, intent(in) :: height, width
    integer, intent(in) :: n, w_len_max, blank
    integer, dimension(:), intent(in) :: oris, is, js, ks
    integer, dimension(:,:), intent(in) :: words
    integer, dimension(:), intent(in) :: w_lens
    integer, dimension(:,:), intent(inout) :: puzzle
    integer, dimension(:,:), intent(inout) :: enable
    integer, dimension(n), intent(out) :: is_used

    integer :: w_len, a, b, i, j, k, ori, placeability
    integer :: is_placeable
    integer :: word(w_len_max)
    logical :: placed

    logical :: is_already_used
    integer :: c, place_count, used_k(n)
    place_count = 0
    used_k = -1
    is_used = -1
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
                    if (k == used_k(c)) then
                        is_already_used = .true.
                        exit
                    end if
                end do
            end if
            if (is_already_used .eqv. .true.) then
                cycle
            end if
            if (is_used(a) == 0) then
                cycle
            end if

            placeability = is_placeable(puzzle, enable, height, width, ori, i, j, word, w_len, blank)    
            is_used(a) = placeability
            ! write(6,*) a, placeability
            if (placeability /= 0) then
                cycle
            end if
            placed = .true.
            place_count = place_count + 1
            used_k(place_count) = k
            if (oris(a) == 0) then
                do b = 1, w_len
                    puzzle(js(a), is(a)+b-1) = word(b) ! 単語配置
                end do
                ! enable更新
                if (is(a)-1 >= 1) then
                    enable(js(a), is(a)-1) = 0
                end if
                if (is(a)+w_len <= width) then
                    enable(js(a), is(a)+w_len) = 0
                end if
            end if
            if (oris(a) == 1) then
                do b = 1, w_len
                    puzzle(js(a)+b-1, is(a)) = word(b) ! 単語配置
                end do
                ! enable更新
                if (js(a)-1 >= 1) then
                    enable(js(a)-1, is(a)) = 0
                end if
                if (js(a)+w_len <= height) then
                    enable(js(a)+w_len, is(a)) = 0
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
    integer, intent(in) :: puzzle(width, height)
    integer, intent(in) :: enable(width, height)
    integer, intent(in) :: blank
    
    integer :: check, cf, ce
    integer :: a, b
    logical :: is_cross_at_a, at_least_1_cross
    
    ! ori:0縦、1横
    ! i:縦の配列番号　y
    ! j:横の配列番号　x
    
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
    
    !---1. 全空白かどうか確認------------------------------------------------
    check = 0
    do b = 1, height
        do a = 1, width
            if (puzzle(a, b) /= blank) check = 1
        end do
    end do
    if (check == 0) then
        is_placeable = 0 !全て空白のため配置可能
        return
    end if

    !---2. 前後に文字があるかどうか--------------------------------------------
    cf = 0
    ce = 0
    if (ori == 0) then ! 縦
        !一文字目上
        if (i == 1) then
            cf = 1
        else
            if (puzzle(j, i-1) == blank) then
                cf = 1
            end if
        end if
        !最終文字下
        if ((i+w_len-1) == height) then
            ce = 1
        else
            if (puzzle(j, i+w_len) == blank) then
                ce = 1
            end if
        end if
    else if (ori == 1) then ! 横
        !一文字目左
        if (j == 1) then
            cf = 1
        else
            if (puzzle(j-1, i) == blank) then
                cf = 1
            end if
        end if
        !最終文字右
        if ((j+w_len-1) == width) then
            ce = 1
        else
            if (puzzle(j+w_len, i) == blank) then
                ce = 1
            end if
        end if
    end if
    if (cf == 0 .or. ce == 0) then
        is_placeable = 1 !前後に文字が存在するため配置不可能
        return
    end if
    
    !---3.他の単語とクロスするか and クロス部分は一致しているか and US/USA問題--------------------
    if (ori == 0) then ! 縦
        at_least_1_cross = .false. !クロス部が存在すれば at_least_1_cross = .true.
        do a = 1-1, w_len-1
            is_cross_at_a = .false. !クロス部が存在すれば is_cross_at_a = .true.
            if (puzzle(j, i+a) /= blank) then
                is_cross_at_a = .true.
                at_least_1_cross = .true.
                if (puzzle(j, i+a) == word(a+1)) then
                    cycle !クロス部の文字が同じなら次の文字へ
                else
                    is_placeable = 3 !クロス部のが同じじゃないため配置不可能
                    return
                end if
            else !クロス部以外のマス
                if (j > 1 .and. puzzle(j-1, i+a) /= blank) then
                    is_placeable = 5 !重なる文字のところ以外の隣のどちらかでも空白でないため配置不可能
                    return
                end if
                if (j < width .and. puzzle(j+1, i+a) /= blank) then
                    is_placeable = 5 !重なる文字のところ以外の隣のどちらかでも空白でないため配置不可能
                    return
                end if
            end if
        end do
        if (at_least_1_cross .eqv. .false.) then
            is_placeable = 2 !クロス部が存在しないため配置不可能
            return
        end if
        do a = 1-1, w_len-1
            if (enable(j, i+a) == 0) then
                is_placeable = 6 ! US/USA問題のため配置不可能
                return
            end if
        end do
    else if (ori == 1) then !横
        at_least_1_cross = .false. !クロス部が存在すれば at_least_1_cross = .true.
        do a = 1-1, w_len-1
            is_cross_at_a = .false.
            if (puzzle(j+a, i) /= blank) then
                is_cross_at_a = .true.
                at_least_1_cross = .true.
                if (puzzle(j+a, i) == word(a+1)) then
                    cycle !クロス部の文字が同じなら次の文字へ
                else
                    is_placeable = 3 !クロス部の文字が同じじゃないため配置不可能
                    return
                end if
            else !クロス部以外のマス
                if (i > 1 .and. puzzle(j+a, i-1) /= blank) then
                    is_placeable = 5 !重なる文字のところ以外の隣のどちらかでも空白でないため配置不可能
                    return
                end if
                if (i < height .and. puzzle(j+a, i+1) /= blank) then
                    is_placeable = 5 !重なる文字のところ以外の隣のどちらかでも空白でないため配置不可能
                    return
                end if
            end if
        end do
        if (at_least_1_cross .eqv. .false.) then
            is_placeable = 2 !クロス部が存在しないため配置不可能
            return
        end if
        do a = 1-1, w_len-1
            if (enable(j+a, i) == 0) then
                is_placeable = 6 !US/USA問題のため配置不可能
                return
            end if
        end do
    end if

    is_placeable = 0
    return
end function