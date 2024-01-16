-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 15 Jan 2024 pada 10.43
-- Versi server: 10.4.27-MariaDB
-- Versi PHP: 8.1.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `review`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `hasil_model`
--

CREATE TABLE `hasil_model` (
  `id_hasil_model` int(11) NOT NULL,
  `id_review` int(11) NOT NULL,
  `nama` varchar(255) NOT NULL,
  `tanggal` date NOT NULL,
  `review` varchar(255) NOT NULL,
  `label` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `hasil_model`
--

INSERT INTO `hasil_model` (`id_hasil_model`, `id_review`, `nama`, `tanggal`, `review`, `label`) VALUES
(90, 121, 'Suep', '2023-12-01', 'Bagus', 5),
(91, 122, 'Verza', '2023-12-02', 'Bagus bangett mantappp', 5),
(92, 123, 'Rosi', '2023-12-03', 'Bagus', 5),
(93, 124, 'Tari', '2023-12-04', 'aplikasinya oke bagus bangettt mantapppp', 5),
(94, 125, 'Tarmo', '2023-12-05', 'aplikasinya oke bagus bangettt mantapppp', 5),
(95, 126, 'Nopal', '2023-12-06', 'aplikasinya oke bagus bangettt mantapppp', 5),
(96, 127, 'Reza', '2023-12-07', 'bagus bangettt mantapppp', 3),
(97, 128, 'Feri', '2023-12-08', 'bagus bangettt mantapppp', 3),
(98, 129, 'Atha', '2023-12-09', 'bagus bangettt mantapppp', 3),
(99, 130, 'Nimah', '2023-12-10', 'good', 5),
(100, 131, 'Torja', '2023-12-11', 'bagus', 5),
(101, 132, 'Suni', '2023-12-12', 'bagus', 5),
(102, 133, 'Yika', '2023-12-13', 'bagus', 5),
(103, 134, 'Kondor', '2023-12-01', 'jelek banget', 1),
(104, 135, 'Rama', '2023-12-02', 'aplikasi jelek banget', 1),
(105, 136, 'Afkar', '2023-12-03', 'aplikasi jelek banget', 1),
(106, 137, 'Rokayah', '2023-12-04', 'sampah, aplikasi jelek banget', 1),
(107, 138, 'Leni', '2023-12-01', 'lumayan sih', 3),
(108, 139, 'Zen', '2023-12-02', 'lumayan', 3),
(109, 140, 'Kori', '2023-12-03', 'lumayan sihh', 3),
(110, 141, 'Rizki', '2023-12-04', 'lumayan sihh', 3),
(111, 142, 'Maulana', '2023-12-05', 'lumayan sihh', 3),
(112, 143, 'Sobri', '2023-12-06', 'lumayan sih', 3),
(113, 144, 'Durma', '2023-12-07', 'lumayan deh', 3),
(114, 145, 'Sonya', '2023-12-08', 'lumayan deh', 3),
(115, 146, 'Kincana', '2023-12-20', 'lumayan', 3),
(116, 146, 'Kincana', '2023-12-20', 'lumayan', 3),
(117, 147, 'Rendi', '2023-12-21', 'Bagus bangett mantappp', 5),
(118, 148, 'Andre', '2023-12-20', 'good', 5),
(119, 149, 'Putri', '2023-12-20', 'good', 5),
(120, 150, 'Zidan', '2023-12-20', 'good', 5),
(121, 151, 'Maul', '2023-12-20', 'good', 5),
(122, 152, 'Ana', '2023-12-20', 'good', 5),
(123, 153, 'Ghani', '2023-12-21', 'jelek banget', 1),
(124, 154, 'Hendra', '2023-12-23', 'Bagus', 3),
(125, 155, 'Sarah', '2023-12-26', 'Bagus bangett mantappp', 3),
(126, 156, 'Yosan', '2023-12-07', 'jelek banget', 3),
(127, 157, 'Rio', '2023-12-12', 'sampah, aplikasi jelek banget', 3),
(128, 158, 'Nami', '2023-12-24', 'good', 3),
(129, 159, 'Feni', '2023-12-20', 'Bagus bangett mantappp', 3),
(130, 160, 'Jos', '2023-12-29', 'good', 5),
(131, 161, 'Bayu', '2023-12-20', 'good!!', 5),
(132, 162, 'Kape', '2023-12-20', 'Bagus bangett mantappp', 5),
(133, 163, 'Dego', '2023-12-20', 'Bagus bangett mantappp', 5),
(134, 164, 'Chori', '2023-12-22', 'Bagus bangett mantappp', 5);

-- --------------------------------------------------------

--
-- Struktur dari tabel `input_review`
--

CREATE TABLE `input_review` (
  `id_review` int(11) NOT NULL,
  `nama` varchar(255) NOT NULL,
  `tanggal` date NOT NULL,
  `review` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `hasil_model`
--
ALTER TABLE `hasil_model`
  ADD PRIMARY KEY (`id_hasil_model`);

--
-- Indeks untuk tabel `input_review`
--
ALTER TABLE `input_review`
  ADD PRIMARY KEY (`id_review`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `hasil_model`
--
ALTER TABLE `hasil_model`
  MODIFY `id_hasil_model` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=135;

--
-- AUTO_INCREMENT untuk tabel `input_review`
--
ALTER TABLE `input_review`
  MODIFY `id_review` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=165;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
