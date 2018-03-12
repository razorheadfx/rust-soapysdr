#[macro_use]
extern crate log;
extern crate num_complex;
extern crate rand;
extern crate soapysdr;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;

use num_complex::Complex;
use structopt::StructOpt;
use std::process;
use std::time::{Duration, SystemTime};
use rand::Rng;
use std::thread;

#[derive(StructOpt, Debug)]
#[structopt(name = "soapy-uhd-test", about = "a soapysdr-uhd test and example app")]
struct Conf {
    #[structopt(name = "DEVICE_TYPE", help = "UHD Device Type (e.g. B200, X300)")]
    dev_type: String,
    #[structopt(short = "m", long = "mode", help = "Receive (rx) or transmit (tx)",
                default_value = "rx")]
    mode: String,
    #[structopt(short = "d", long = "duration", help = "Duration in seconds", default_value = "5")]
    dur: usize,
    #[structopt(short = "s", long = "srate", help = "Samplerate to use in Hz",
                default_value = "1e6")]
    srate: f64,
    #[structopt(short = "f", long = "frequency", help = "Channel frequency",
                default_value = "2410e6")]
    /// defaults to ISM 2.4GHz band
    freq: f64,
    #[structopt(short = "c", long = "channel", help = "Channel number", default_value = "0")]
    channel: usize,
    #[structopt(short = "bw", long = "bandwidth", help = "Bandwidth", default_value = "1e6")]
    bw: f64,
    #[structopt(short = "g", long = "gain", help = "Gain in db", default_value = "0")]
    gain: f64,
}

// convenience macro
macro_rules! config_channel{
    ($dev:ident, $chan:expr, $dir:expr, $freq:expr, $srate:expr, $bw:expr, $gain:expr) =>({
        $dev.set_sample_rate($dir, $chan, $srate).expect("Failed to set sample Rate");
        $dev.set_frequency($dir, $chan, $freq, "").expect("Failed to set frequency");
        $dev.set_bandwidth($dir, $chan, $bw).expect("Failed to  set banwidth");
        $dev.set_gain($dir, $chan, $gain).expect("Failed to set gain");
        println!("\tConfigured Channel {}:{:?} for srate:{} MSPS, freq:{} MHz, bw:{} MHz, gain:{}dB", $chan, $dir, $srate/1e6, $freq/1e6, $bw/1e6, $gain);
        println!("\tActually got:              srate:{} MSPS, freq:{} MHz, bw:{} MHz, gain:{}dB",
            $dev.sample_rate($dir, $chan)?/1e6,
            $dev.frequency($dir, $chan)?/1e6,
            $dev.bandwidth($dir, $chan)?/1e6,
            $dev.gain($dir, $chan)?
        );
    });
}

// TODO: might add digest (run x, success y, failures ) at the end
macro_rules! run_check{
	($name:expr,$check:expr) => {
		match $check{
			Ok(_) => println!("\t{} check success!", $name),
			Err(e) => println!("\t{} check failed! Reason: {}",$name, e)
		}

	}
}

fn main() {
    let conf = Conf::from_args();

    match run(&conf) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("An error occured: {}", e);
            process::exit(1)
        }
    }
}

fn run(conf: &Conf) -> Result<(), soapysdr::Error> {
    //find the device we want
    let mut filter = soapysdr::Args::new();
    filter.set("type", conf.dev_type.clone());

    // this filter the args and pick the first set of device args that matches
    let arg_set = soapysdr::enumerate(filter)
        .expect("Failed to enumerate devs")
        .into_iter()
        .nth(0)
        .expect(&format!(
            "Failed to find device: type={}. Is that the correct device type?",
            conf.dev_type
        ));

    println!("** Opening Device ");
    let mut d = soapysdr::Device::new(arg_set)?;

    println!(
        "Device has hardware properties: {}",
        d.hardware_info().expect("Failed to get hardware infos"),
    );
    //wait some time for dev init and checks (for good measure)
    //UHD is quite spammy in that regard
    thread::sleep(Duration::from_millis(500));

    run_check!("Timing", check_timing(&mut d, &conf));

    run_check!("Continuous RX", check_continuous_reception(&mut d, &conf));

    //run_check!("Burst RX", check_burst_reception(&mut d, &conf));

    run_check!("Continuous TX", check_continuous_transmission(&mut d, &conf));

    run_check!("Burst TX", check_burst_transmission(&mut d, &conf));


    // TODO: multiple stream formats, not just cf32

    // TODO: multiple stream formats, not just cf32

    // TODO: impl read_burst in Device.rs

    // TODO: impl Log level switching

    Ok(())
}

fn check_timing(d: &mut soapysdr::Device, _conf: &Conf) -> Result<(), soapysdr::Error> {
    println!("** Checking time related functions");

    println!("\tListing HW Time Sources");
    // do we have a hw time source?
    if d.has_hardware_time(None)
        .expect("This device cannot indicate whether it has time sources?!")
    {
        // lets enumerate
        for source in d.list_time_sources().expect("Failed to list time sources") {
            println!(
                "\tListed time source: {:?} Timestamp: {}",
                &source,
                d.get_hardware_time(Some(source.as_str())).expect(&format!(
                    "Failed to retrieve hardware time from {:?}",
                    source
                ))
            );
        }
        // if they only differ by a few thousand ns
        // you can be sure that you are getting the same time
    }

    // actually UHDs only support 2 hw timestamp sources with different hw times
    let uhd_time_sources = [None, Some("PPS")].iter();

    println!("\tChecking UHD supported hardware time sources");
    for hw_time_source in uhd_time_sources {
        let t = d.get_hardware_time(*hw_time_source).expect(&format!(
            "Failed to get Actual HW timestamp from Clock Source {:?}",
            *hw_time_source
        ));
        println!("\tActual HW timestamp from {:?} is {}", hw_time_source, t);

        println!("\tResetting time source: {:?}", hw_time_source);

        d.set_hardware_time(*hw_time_source, 0)
            .expect("Failed to reset time source");

        // since we do not know how long it will take to retrieve the new timestamp from the device
        // we just check whether it is lower
        let mut t_new = d.get_hardware_time(*hw_time_source)?;
        debug!(
            "Time elapsed since reset of {:?} : {}",
            hw_time_source, t_new
        );

        // PPS source cannot be reset
        if !hw_time_source.eq(&Some("PPS")) {
            assert!(t_new < t);
        }
    }

    Ok(())
}

fn check_continuous_reception(
    d: &mut soapysdr::Device,
    conf: &Conf,
) -> Result<(), soapysdr::Error> {
    let channel = conf.channel;

    // you need need to configure the channel RF components first
    // at least essentials such as frequency and sample rate
    // if you fail to set these beforehand we are not guaranteed to
    // get an error somewhere here, it will fail silently and we'll
    // wonder why no samples are coming in once we start calling RxStream::read()
    // in the other check methods we'll be using the config_channel! macro
    // for convenience
    let dir = soapysdr::Direction::Rx;
    d.set_sample_rate(dir, channel, conf.srate).expect("Failed to set sample rate");
    // the device may set the actual frequency to something different subject to
    // the frequency range available to the local clock source
    // lets see how close we got
    let actual_srate = d.sample_rate(dir, channel).expect("Failed to retrieve sample rate");
    println!("\tTarget sample rate: {:.3}MSPS, actual: {:.3}MSPS, deviation: {:.3}MSPS",
        conf.srate,
        actual_srate,
        actual_srate-conf.srate
    );
    // same goes for frequency, bandwidth and gain
    let args = "";
    d.set_frequency(dir, channel, conf.freq, args).expect("Failed to set frequency");
    let freq = d.frequency(dir, channel).expect("Failed to retrieve frequency");
    println!("\tTarget frequency: {:.3}MHz, actual: {:.3}MHz, deviation: {:.3}MHz",
        conf.freq,
        freq,
        freq-conf.freq
    );

    d.set_bandwidth(dir, channel, conf.bw).expect("Failed to set bwndwidth");
    let bw = d.bandwidth(dir, channel).expect("Failed to retrieve bandwidth");
    println!("\tTarget bandwidth: {:.3}MHz, actual: {:.3}MHz, deviation {:.3}MHz",
        conf.bw,
        bw,
        bw-conf.bw
    );

    d.set_gain(dir, channel, conf.gain).expect("Failed to set gain");
    let gain = d.gain(dir, channel).expect("Failed to retrieve gain");
    println!("\tTarget bandwidth: {}dB, actual: {}dB, deviation {}dB",
        conf.gain,
        gain,
        gain-conf.gain
    );

    // open the channel for reading
    let channels_to_open_for_reading = [conf.channel];
    let mut rx = d.rx_stream::<Complex<f32>>(&channels_to_open_for_reading)
        .expect("Failed to open the channel for reception");

    // lets get the maximum transmission unit for the stream
    // from device to host and build a buffer of that size
    let mtu = rx.mtu().expect("Failed to retrieve MTU");
    println!("\tStream MTU is {}", mtu);

    let mut rx_samples = vec![Complex::<f32>::default(); mtu];

    let dur = Duration::from_secs(conf.dur as u64);
    let start = SystemTime::now();
    
    // activate the stream and read some samples
    println!("\tActivating stream and reading samples for {}s",conf.dur);

    let mut rx_acc = 0usize;

    rx.activate(None).expect("Failed to activate the stream");
    while start.elapsed().unwrap() < dur{
        let rx_ed = rx.read(&[&mut rx_samples], 100_000_000)
            .expect("Failed to read samples");
        assert_eq!(rx_ed, mtu);
	rx_acc+=rx_ed;
    }
    // deactivate the stream
    rx.deactivate(None).expect("Failed to deactivate stream");
    
    println!("\tDone, stream deactivated after reading {} samples",rx_acc);    

    Ok(())


}

fn check_burst_reception(
    _d: &mut soapysdr::Device,
    _conf: &Conf,
) -> Result<(), soapysdr::Error> {
	unimplemented!()
}

fn check_continuous_transmission(
    d: &mut soapysdr::Device,
    conf: &Conf,
) -> Result<(), soapysdr::Error> {
    let dur = Duration::from_secs(conf.dur as u64);
    println!(
        "** Checking continuous transmission by sending noise for {}s",
        dur.as_secs()
    );

    // TODO: move this comment to first calls to set_freq etc

    config_channel!(
        d,
        conf.channel,
        soapysdr::Direction::Tx,
        conf.freq,
        conf.srate,
        conf.bw,
        conf.gain
    );

    let channel_list = [conf.channel];
    let mut tx = d.tx_stream::<Complex<f32>>(&channel_list)
        .expect("Failed to open stream");

    let mtu = tx.mtu().expect("Failed to get stream MTU");
    // fill buffer with noise
    // for simplicities sake we're not gonna generate a new one
    // for every call to tx
    let noise = make_noise(mtu);

    let start = SystemTime::now();

    //activate immediately
    tx.activate(None)
        .expect("Failed to activate TxStream immediately");

    println!("\tStarting Transmission");
    let mut acc = 0;
    while start.elapsed().unwrap() < dur {
        let txed = tx.write(&[&noise], None, false, 100_000_000)
            .expect("Failed to write to stream immediately)");
        assert_eq!(txed, noise.len());
        acc += txed;
    }
    tx.deactivate(None)
        .expect("Failed to deactivate TxStream immediatly");

    println!("\tTransmitted {} samples", acc);
    // TODO: activate at time and check it blocks until tx time
    // by checking host timestamp after the call

    Ok(())
}

fn check_burst_transmission(d: &mut soapysdr::Device, conf: &Conf) -> Result<(), soapysdr::Error> {
    let burst_dur = Duration::from_millis(5);
    println!(
        "** Checking burst transmission writing 10 bursts of length {}ms to TxStream",
        burst_dur.subsec_nanos() / 1_000_000
    );
    // write 10 bursts with 2*burst_dur ms inter-transmission time

    config_channel!(
        d,
        conf.channel,
        soapysdr::Direction::Tx,
        conf.freq,
        conf.srate,
        conf.bw,
        conf.gain
    );

    let mut tx = d.tx_stream::<Complex<f32>>(&[conf.channel])
        .expect("Failed to open stream");

    let samp_dur_ns = (d.sample_rate(soapysdr::Direction::Tx, conf.channel)?
        .recip() * 1e9f64) as u32; // duration of 1 sample
    let samps_x_burst = burst_dur.subsec_nanos() / samp_dur_ns; // samples per burst
    let noise = make_noise(samps_x_burst as usize); // a buffer of noise
    println!("\tAt this sample rate samples are {}ns long", samp_dur_ns);
    println!(
        "\tBursts will be {}ns ({} Samples) long (par: {}ns) and be spaced {}ns apart",
        samps_x_burst * samp_dur_ns,
        samps_x_burst,
        burst_dur.subsec_nanos(),
        burst_dur.subsec_nanos()
    );

    // get ourself a relative time reference
    let ref_time_ns = d.get_hardware_time(None)
        .expect("Failed to get reference timestamp");
    let timeout = Duration::from_millis(100).subsec_nanos() as i64;
    // start the first burst after waiting an initial 5ms
    let mut next_tx = ref_time_ns + Duration::from_millis(5).subsec_nanos() as i64;

    // transmit bursts in a loop
    let mut burst_no = 0;
    while burst_no < 10 {
    	// write_all will write the entire burst in one call
        tx.write_all(&[&noise], Some(next_tx), true, timeout)
            .expect("Transmitting burst failed");
        //start the next burst 1 burst duration after this one is done
        //i.e. current txtime + 2 times burst duration
        next_tx += (burst_dur * 2).subsec_nanos() as i64;
        burst_no += 1;
    }

    // TODO: implement read_stream_status for tx stream
    // right now we can only check if there were any
    // - late (L)
    // - out of sequence transmission (S)
    // - TX buffer underflows (U)
    // Events by checking whether UHD printed something to standard out here.

    Ok(())
}



fn make_noise(len: usize) -> Vec<Complex<f32>> {
    rand::thread_rng()
        .gen_iter::<(f32, f32)>()
        .take(len)
        .map(|c| Complex { re: c.0, im: c.1 })
        .collect()
}
