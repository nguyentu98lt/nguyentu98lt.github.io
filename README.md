# Bài 5: Giới thiệu về xử lý ảnh \| Deep Learning cơ bản

## Ảnh trong máy tính

### Hệ màu RGB

RGB viết tắt của red \(đỏ\), green \(xanh lục\), blue \(xanh lam\), là ba màu chính của ánh sáng khi tách ra từ lăng kính. Khi trộn ba màu trên theo tỉ lệ nhất định có thể tạo thành các màu khác nhau.![](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/RGB.png?w=1400&ssl=1)

Thêm đỏ vào xanh lá cây tạo ra vàng; thêm vàng vào xanh lam tạo ra trắng. Nguồn [wiki](https://vi.wikipedia.org/wiki/M%C3%B4_h%C3%ACnh_m%C3%A0u_RGB#/media/File:AdditiveColorMixiing.svg).

Ví dụ khi bạn chọn màu ở [đây](https://www.w3schools.com/colors/colors_picker.asp). Khi bạn chọn một màu thì sẽ ra một bộ ba số tương ứng **\(r,g,b\)**![](https://i1.wp.com/nttuan8.com/wp-content/uploads/2019/03/RGB-picker.png?resize=524%2C426&ssl=1) 

màu được chọn là rgb\(102, 255, 153\), nghĩa là r=102, g=255, b=153.

Với mỗi bộ 3 số r, g, b nguyên trong khoảng \[0, 255\] sẽ cho ra một màu khác nhau. Do có 256 cách chọn r, 256 cách chọn màu g, 256 cách chọn b =&gt; tổng số màu có thể tạo ra bằng hệ màu RGB là: 256 \* 256 \* 256 = 16777216 màu !!!

### Ảnh màu

Ví dụ về ảnh màu

![](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/mathematical-bridge.jpg?w=1400&ssl=1)

Mathematical bridge, Cambridge

Khi bạn kích chuột phải vào ảnh trong máy tính, bạn chọn properties \(mục cuối cùng\), rồi chọn tab detail![](https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/03/image_property.png?w=1400&ssl=1)

Bạn sẽ thấy chiều dài ảnh là 800 pixels \(viết tắt px\), chiều rộng 600 pixels, kích thước là 800 \* 600. Trước giờ chỉ học đơn vị đo là mét hay centimet, pixel là gì nhỉ ?

Theo [wiki](https://vi.wikipedia.org/wiki/Pixel), pixel \(hay điểm ảnh\) là một khối màu rất nhỏ và là đơn vị cơ bản nhất để tạo nên một bức ảnh kỹ thuật số.

Vậy bức ảnh trên kích thước 800 pixel \* 600 pixel, có thể biểu diễn dưới dạng một [ma trận](https://nttuan8.com/bai-1-linear-regression-va-gradient-descent/#Ma_tran) kích thước 600 \* 800 \(vì định nghĩa ma trận là số hàng nhân số cột\).

![](https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/03/CodeCogsEqn-9-1.gif?w=1400&ssl=1)

Trong đó mỗi phần tử w\_{ij} là một pixel.

Như vậy có thể hiểu là mỗi pixel thì biểu diễn một màu và bức ảnh trên là sự kết hợp rất nhiều pixel. Hiểu đơn giản thì in bức ảnh ra, kẻ ô vuông như chơi cờ ca rô với 800 đường thẳng ở chiều dài, 600 đường ở chiều rộng, thì mỗi ô vuông là một pixel, biểu diễn một chấm màu.

Tuy nhiên để biểu diễn 1 màu ta cần 3 thông số \(r,g,b\) nên gọi w\_{ij} = \(r\_{ij}, g\_{ij}, b\_{ij}\) để biểu diễn dưới dạng ma trận thì sẽ như sau:

![](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/CodeCogsEqn-4-2.gif?w=1400&ssl=1)

Ảnh màu kích thước 3\*3 biểu diễn dạng ma trận, mỗi pixel biểu diễn giá trị \(r,g,b\)  


Để tiện lưu trữ và xử lý không thể lưu trong 1 ma trận như thế kia mà sẽ tách mỗi giá trị trong mỗi pixel ra một ma trận riêng.

![](https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/03/CodeCogsEqn-5-3.gif?w=1400&ssl=1)

Tách ma trận trên thành 3 ma trận cùng kích thước: mỗi ma trận lưu giá trị từng màu khác nhau red, green, blue

Tổng quát

![](https://i1.wp.com/nttuan8.com/wp-content/uploads/2019/03/rgb.gif?w=1400&ssl=1)

Tách ma trận biểu diễn màu ra 3 ma trận, mỗi ma trận lưu giá trị 1 màu.

Mỗi ma trận được tách ra được gọi là 1 channel nên ảnh màu được gọi là 3 channel: channel red, channel green, channel blue.

**Tóm tắt**: Ảnh màu là một ma trận các pixel mà mỗi pixel biểu diễn một điểm màu. Mỗi điểm màu được biểu diễn bằng bộ 3 số \(r,g,b\). Để tiện cho việc xử lý ảnh thì sẽ tách ma trận pixel ra 3 channel red, green, blue.

### Tensor là gì

Khi dữ liệu biểu diễn dạng 1 chiều, người ta gọi là vector, mặc định khi viết vector sẽ viết dưới dạng cột.

Khi dữ liệu dạng 2 chiều, người ta gọi là ma trận, kích thước là số hàng \* số cột.![](https://i1.wp.com/nttuan8.com/wp-content/uploads/2019/03/CodeCogsEqn-8-1.gif?w=1400&ssl=1)

Vector v kích thước n, ma trận W kích thước m\*n  


Khi dữ liệu nhiều hơn 2 nhiều thì sẽ được gọi là tensor, ví dụ như dữ liệu có 3 chiều.

Để ý thì thấy là ma trận là sự kết hợp của các vector cùng kích thước. Xếp n vector kích thước m cạnh nhau thì sẽ được ma trận m\*n. Thì tensor 3 chiều cũng là sự kết hợp của các ma trận cùng kích thước, xếp k ma trận kích thước m\*n lên nhau sẽ được tensor kích thước m\*n\*k.![](https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/03/H%C3%ACnh_h%E1%BB%99p_ch%E1%BB%AF_nh%E1%BA%ADt-1.png?resize=430%2C256&ssl=1)

Hình hộp chữ nhật kích thước a\*b\*h

Tưởng tượng mặt đáy là một ma trận kích thước a \* b, được tạo bởi b vector kích thước a. Cả hình hộp là tensor 3 chiều kích thước a\*b\*h, được tạo bởi xếp h ma trận kích thước a\*b lên nhau.

Do đó biểu diễn ảnh màu trên máy tính ở phần trên sẽ được biểu diễn dưới dạng tensor 3 chiều kích thước 600\*800\*3 do có 3 ma trận \(channel\) màu red, green, blue kích thước 600\*800 chồng lên nhau.

Ví dụ biểu diễn ảnh màu kích thước 28\*28, biểu diễn dưới dạng tensor 28\*28\*3![](https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/03/tensor.jpg?resize=568%2C426&ssl=1)

Nguồn: https://www.slideshare.net/BertonEarnshaw/a-brief-survey-of-tensors

### Ảnh xám

![](https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/03/gray.jpg?w=1400&ssl=1)

Ảnh xám của mathematical bridge



Tương tự ảnh màu, ảnh xám cũng có kích thước 800 pixel \* 600 pixel, có thể biểu diễn dưới dạng một [ma trận](https://nttuan8.com/bai-1-linear-regression-va-gradient-descent/#Ma_tran) kích thước 600 \* 800 \(vì định nghĩa ma trận là số hàng nhân số cột\).![](https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/03/CodeCogsEqn-9-1.gif?w=1400&ssl=1)



Tuy nhiên mỗi pixel trong ảnh xám chỉ cần biểu diễn bằng một giá trị nguyên trong khoảng từ \[0,255\] thay vì \(r,g,b\) như trong ảnh màu. Do đó khi biểu diễn ảnh xám trong máy tính chỉ cần một ma trận là đủ.![](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/CodeCogsEqn-10-1.gif?w=1400&ssl=1)

Biểu diễn ảnh xám



Giá trị 0 là màu đen, 255 là màu trắng và giá trị pixel càng gần 0 thì càng tối và càng gần 255 thì càng sáng.

### 

### Chuyển hệ màu của ảnh

Mỗi pixel trong ảnh màu được biểu diễn bằng 3 giá trị \(r,g,b\) còn trong ảnh xám chỉ cần 1 giá trị x để biểu diễn.

Khi chuyển từ ảnh màu sang ảnh xám ta có thể dùng [công thức](https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert): x = r \* 0.299 + g \* 0.587 + b \* 0.114.

Tuy nhiên khi chuyển ngược lại, bạn chỉ biết giá trị x và cần đi tìm r,g,b nên sẽ không chính xác.

## 

Phép tính convolution

### 

### Convolution

Để cho dễ hình dung mình sẽ lấy ví dụ trên ảnh xám, tức là ảnh được biểu diễn dưới dạng ma trận A kích thước m\*n.

Ta định nghĩa **kernel** là một ma trận vuông kích thước k\*k trong đó k là số lẻ. k có thể bằng 1, 3, 5, 7, 9,… Ví dụ kernel kích thước 3\*3

![](https://i1.wp.com/nttuan8.com/wp-content/uploads/2019/03/CodeCogsEqn-13-1.gif?w=1400&ssl=1)

Kí hiệu phép tính convolution \(\otimes\), kí hiệu Y = X \otimes W

Với mỗi phần tử x\_{ij} trong ma trận X lấy ra một ma trận có kích thước bằng kích thước của kernel W có phần tử x\_{ij} làm trung tâm \(đây là vì sao kích thước của kernel thường lẻ\) gọi là ma trận A. Sau đó tính tổng các phần tử của phép tính [element-wise](https://nttuan8.com/bai-1-linear-regression-va-gradient-descent/#Element-wise_multiplication_matrix) của ma trận A và ma trận W, rồi viết vào ma trận kết quả Y.![](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/c1.png?resize=543%2C251&ssl=1)

Ví dụ khi tính tại x\_{22} \(ô khoanh đỏ trong hình\), ma trận A cùng kích thước với W, có x\_{22} làm trung tâm có màu nền da cam như trong hình. Sau đó tính y\_{11} = sum\(A \otimes W\) = x\_{11}\*w\_{11} + x\_{12}\*w\_{12} + x\_{13}\*w\_{13} + x\_{21}\*w\_{21} + x\_{22}\*w\_{22} + x\_{23}\*w\_{23} + x\_{31}\*w\_{31} + x\_{32}\*w\_{32} + x\_{33}\*w\_{33} = 4. Và làm tương tự với các phần tử còn lại trong ma trận.

Thế thì sẽ xử lý thế nào với phần tử ở viền ngoài như x\_{11}? Bình thường khi tính thì sẽ bỏ qua các phần tử ở viền ngoài, vì không tìm được ma trận A ở trong X

![](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/c2.png?resize=294%2C263&ssl=1)

Nên bạn để ý thấy ma trận Y có kích thước nhỏ hơn ma trận X. Kích thước của ma trận Y là \(m-k+1\) \* \(n-k+1\).

![](https://i1.wp.com/nttuan8.com/wp-content/uploads/2019/03/giphy.gif?w=1400&ssl=1)

Các bước thực hiện phép tính convolution cho ma trận X với kernel K ở trên

### Padding

Như ở trên thì mỗi lần thực hiện phép tính convolution xong thì kích thước ma trận Y đều nhỏ hơn X. Tuy nhiên giờ ta muốn ma trận Y thu được có kích thước bằng ma trận X =&gt; Tìm cách giải quyết cho các phần tử ở viền =&gt; Thêm giá trị 0 ở viền ngoài ma trận X.

![](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/c3.png?resize=374%2C348&ssl=1)

Ma trận X khi thêm viền 0 bên ngoài  


Rõ ràng là giờ đã giải quyết được vấn đề tìm A cho phần tử x\_{11} , và ma trận Y thu được sẽ bằng kích thước ma trận X ban đầu.

Phép tính này gọi là convolution với **padding=1**. Padding=k nghĩa là thêm k vector 0 vào mỗi phía của ma trận.

### Stride

Như ở trên ta thực hiện tuần tự các phần tử trong ma trận X, thu được ma trận Y cùng kích thước ma trận X, ta gọi là stride=1

![](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/c4.png?resize=383%2C357&ssl=1) 

stride=1, padding=1

Tuy nhiên nếu **stride=k** \(k &gt; 1\) thì ta chỉ thực hiện phép tính convolution trên các phần tử x\_{1+i\*k,1+j\*k}. Ví dụ k = 2.

![](https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/03/c5.png?resize=378%2C373&ssl=1)

padding=1, stride=2

Hiểu đơn giản là bắt đầu từ vị trí x\_{11} sau đó nhảy k bước theo chiều dọc và ngang cho đến hết ma trận X.

Kích thước của ma trận Y là 3\*3 đã giảm đi đáng kể so với ma trận X.

Công thức tổng quát cho phép tính convolution của ma trận X kích thước m\*n với kernel kích thước k\*k, stride = s, padding = p ra ma trận Y kích thước \displaystyle\(\frac{m-k+2p}{s}+1\) \* \(\frac{n-k+2p}{s}+1\).

Stride thường dùng để giảm kích thước của ma trận sau phép tính convolution.

Mọi người có thể xem thêm trực quan hơn ở [đây](https://github.com/vdumoulin/conv_arithmetic).

### Ý nghĩa của phép tính convolution

Mục đích của phép tính convolution trên ảnh là làm mở, làm nét ảnh; xác định các đường;… Mỗi kernel khác nhau thì sẽ phép tính convolution sẽ có ý nghĩa khác nhau. Ví dụ:![](https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/03/purpose.png?resize=527%2C653&ssl=1)



Nguồn: https://en.wikipedia.org/wiki/Kernel\_\(image\_processing\)

   


